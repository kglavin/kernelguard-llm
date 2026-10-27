"""HF runtime tripwire (P5).

Forward hook on decoder block L-1 (gives hidden_states[L]) scores the newest token position of
every sequence against the probe bank on device, updates an EMA on device, and sets a per-sequence
violation flag. A LogitsProcessor forces EOS for flagged sequences. No host sync in the hot path.
Scores are copied to host only after generation for reporting.

Gate rule (Rev 2.0 §10):
  fast trip       s_t        >= tau_crit  (tau + crit_margin)
  sustained trip  ema_t      >= tau       for k consecutive steps

With bank_out set, the prefill step (step 0, last prompt token) is scored by the inbound bank with
a plain s >= tau rule, and every decode step is scored by the outbound bank with the gate rule above.
out_mode='token' scores each generated token's state. out_mode='mean' scores the running mean of the
generated tokens' states so far, which matches an outbound bank trained on response means.
Without bank_out, one bank scores every step (the S0 behaviour).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from kernelguard.extract.activations import ModelBundle
from kernelguard.probes.bank import ProbeBank


@dataclass
class TripState:
    scores: list[torch.Tensor] = field(default_factory=list)  # per step [B, P]
    ema: torch.Tensor | None = None  # [B, P]
    streak: torch.Tensor | None = None  # [B, P] int
    flagged: torch.Tensor | None = None  # [B] bool
    trip_step: torch.Tensor | None = None  # [B] int, -1 if none
    step: int = 0
    out_sum: torch.Tensor | None = None  # [B, d] running sum of decode-step states (out_mode='mean')
    out_n: int = 0


class Tripwire:
    def __init__(
        self,
        mb: ModelBundle,
        bank: ProbeBank,
        layer: int,
        alpha: float = 0.6,
        k_sustained: int = 2,
        crit_margin: float = 0.2,
        bank_out: ProbeBank | None = None,
        out_mode: str = "token",
    ):
        if out_mode not in ("token", "mean"):
            raise ValueError("out_mode must be 'token' or 'mean'")
        self.out_mode = out_mode
        self.mb = mb
        self.layer = layer
        self.alpha = alpha
        self.k = k_sustained
        self.crit_margin = crit_margin
        dev, dt = mb.device, next(mb.model.parameters()).dtype

        def put(b: ProbeBank) -> dict:
            return {
                "mu": torch.tensor(b.mu, device=dev, dtype=dt),
                "sigma": torch.tensor(b.sigma, device=dev, dtype=dt),
                "W": torch.tensor(b.W, device=dev, dtype=dt),  # [P, d]
                "tau": torch.tensor(b.tau, device=dev, dtype=dt),  # [P]
            }

        self.inp = put(bank)
        self.out = put(bank_out) if bank_out is not None else None
        self.names = list(bank.names)
        self.state = TripState()
        self._handle = None

    # ---- hook ----------------------------------------------------------
    @staticmethod
    def _score(x: torch.Tensor, P: dict) -> torch.Tensor:
        x = (x - P["mu"]) / P["sigma"]
        x = torch.nn.functional.normalize(x, dim=-1, eps=1e-6)
        return x @ P["W"].T  # [B, P]

    def _hook(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output  # [B, T, d]
        x = h[:, -1, :]
        st = self.state
        if st.flagged is None:
            st.flagged = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
            st.trip_step = torch.full((x.shape[0],), -1, dtype=torch.int32, device=x.device)
        if self.out is not None and st.step == 0:
            s = self._score(x, self.inp)
            new = (s >= self.inp["tau"]).any(-1) & ~st.flagged
        else:
            P = self.out if self.out is not None else self.inp
            if self.out is not None and self.out_mode == "mean":
                st.out_sum = x.float() if st.out_sum is None else st.out_sum + x.float()
                st.out_n += 1
                x = (st.out_sum / st.out_n).to(x.dtype)
            s = self._score(x, P)
            if st.ema is None:
                st.ema = s.clone()
                st.streak = torch.zeros_like(s, dtype=torch.int32)
            else:
                st.ema = self.alpha * s + (1 - self.alpha) * st.ema
            over = st.ema >= P["tau"]
            st.streak = torch.where(over, st.streak + 1, torch.zeros_like(st.streak))
            fast = (s >= P["tau"] + self.crit_margin).any(-1)
            sustained = (st.streak >= self.k).any(-1)
            new = (fast | sustained) & ~st.flagged
        st.trip_step = torch.where(new, torch.full_like(st.trip_step, st.step), st.trip_step)
        st.flagged = st.flagged | new
        st.scores.append(s.detach())
        st.step += 1

    def arm(self) -> None:
        self.state = TripState()
        block = self.mb.layers[self.layer - 1]
        self._handle = block.register_forward_hook(self._hook)

    def disarm(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    # ---- logits processor ----------------------------------------------
    def logits_processor(self, eos_id: int) -> LogitsProcessor:
        tw = self

        class _Force(LogitsProcessor):
            def __call__(self, input_ids, scores):
                f = tw.state.flagged
                if f is None or not f.any():
                    return scores
                masked = torch.full_like(scores, float("-inf"))
                masked[:, eos_id] = 0.0
                return torch.where(f[:, None], masked, scores)

        return _Force()

    # ---- generation ----------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt_text: str, max_new_tokens: int = 40, **kw) -> dict:
        tok, model = self.mb.tokenizer, self.mb.model
        enc = tok(prompt_text, return_tensors="pt").to(self.mb.device)
        eos = tok.eos_token_id
        self.arm()
        try:
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=eos,
                logits_processor=LogitsProcessorList([self.logits_processor(eos)]),
                **kw,
            )
        finally:
            self.disarm()
        new = out[0, enc.input_ids.shape[1]:]
        st = self.state
        scores = torch.stack(st.scores)[:, 0, :].float().cpu().numpy() if st.scores else None
        trip = int(st.trip_step[0].item()) if st.trip_step is not None else -1
        return {
            "text": tok.decode(new, skip_special_tokens=True),
            "n_new_tokens": int(new.numel()),
            "tripped": trip >= 0,
            "trip_step": trip,  # 0 = prefill (zero tokens emitted), t = after t decode steps
            "scores": scores,  # [steps, P]
            "policies": self.names,
        }
