# KernelGuard-LLM

Research instrument for representation-aware runtime guardrails. Plan: [PROJECT-PLAN.md](PROJECT-PLAN.md).

```
make setup     # uv venv, CPU torch
make test      # unit tests, no model download
make smoke     # S0 end-to-end on the 50 seeds, Qwen2.5-0.5B on CPU
make dataset   # build data/v1/items.jsonl from seeds and templates
make s2 s2-eval s3 s4              # full run on the CPU config
make s2 s3 s4 CONFIG=configs/qwen3b.yaml   # full run on a GPU
```

Results: [reports/FINDINGS.md](reports/FINDINGS.md).

## GPU host

`pyproject.toml` points torch at the CPU-only index. On a CUDA machine, change the index URL under `[tool.uv]` to the
build that matches the driver, for example `https://download.pytorch.org/whl/cu124`, then run `make setup`.
Check with `uv run python -c "import torch; print(torch.cuda.is_available())"`.
The GPU configs batch extraction 16 prompts at a time and write to their own `reports/<model>/` folder.

Layout follows P1 in the plan. Layer indexing: `hidden_states[l]` is the residual stream after block `l` (1-based); a hook for layer `l` registers on `layers[l-1]`.
