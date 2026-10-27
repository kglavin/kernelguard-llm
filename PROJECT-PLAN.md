# KernelGuard-LLM: Plan

Source: [raw-proposal.md](raw-proposal.md). Status: v4, 2026-10-27. S0 to S5 done on Qwen2.5-0.5B, see [reports/FINDINGS.md](reports/FINDINGS.md). Qwen2.5-3B run on the GPU machine is next.

This file is the record of truth for the project. Read it before changing data, code or configs.

## Thesis

Tap one layer of an open-weight LLM on the way in (prompt) and on the way out (response). Learn a partition of that vector space into good and bad from labeled examples, using classical ML on the vectors. Show the partition classifies held-out requests and responses without parsing text.

## Roles

- The LLM is a frozen feature extractor. One forward pass per example yields one vector at layer L.
- The partition is a scikit-learn model on those vectors. Training takes seconds. Inference is a dot product applied inside the forward pass.

## Steps

- S0 Scaffold (done). Schema, extraction, probe bank, runtime hook, metrics, CPU smoke on Qwen2.5-0.5B. `make smoke`.
- S1 Data (done). Enumerated requests labeled good or bad, built from seeds and templates, split by template family. Bad is a request to execute something harmful. Good is benign execution requests and questions about harmful or benign topics. See Data workflow. `make dataset`.
- S2 In (done, 0.5B). Vector at the last prompt token. Logistic regression per policy, layer chosen by leave-one-family-out on the training families, two families per policy held out. Scores bad against all good and bad against benign execution only. `make s2` extracts, `make s2-eval` scores.
- S3 Out (done, 0.5B). One phrasing per request is answered by the model. The vector is the mean over generated tokens at the S2 layer. A response is bad when the request was bad and the model made a tool call. Also calibrates a per-token gate and a running-mean gate, and runs a runtime demo with the inbound bank at prefill and the outbound bank on generated tokens. `make s3`.
- S4 Comparison (done, 0.5B). TF-IDF logistic classifier on request text and on response text, same splits and threshold rule. `make s4`.
- S5 Result (0.5B note done, 3B pending). [reports/FINDINGS.md](reports/FINDINGS.md).
- External check (done, 0.5B). XSTest scored with the S2 partition unchanged. `make xstest`.

## Fixed choices

- Model: Qwen2.5-0.5B-Instruct on CPU for development, Qwen2.5-3B-Instruct on the workstation GPU for the result.
- Bad means a malicious user request. Prompt injection via documents is out of scope.
- Policies: shell exec, credential exfil, SQL mutation.
- Output: internal findings note, no paper.

## Success

S2 or S3 reaching held-out AUROC at or above 0.95 on at least two policies confirms the thesis for this model. Lower numbers are still the result. Either way S5 ships.

## Data workflow

The person running the project writes the bad requests. Claude does not.

- **Bad requests (`exec_mal`).** Written by hand into a file under `data/templates/incoming/`, using the same layout as a template. Claude merges them with `make merge FILE=<file>` and must not type, paste or reword their text. Claude's attempts to write that content were stopped by a safety classifier several times, and the merge command exists so the text never passes through Claude.
- **Benign execution and discussion items.** Claude may write these. They go in an incoming file and are merged the same way, so hand-edited lists stay untouched.
- **Before merging.** Run `make merge FILE="<file> --dry-run"`. It warns on likely typos, such as a missing space before a `$` variable, and refuses families the template does not have.
- **Families.** A new family must be added to the template before items for it can be merged. A family with no bad items is skipped by `make dataset`, so the split always has both labels.
- **Discussion items.** Do not end them with "prose only" or similar markers. Only one discussion wrapper carries that cue, by design, so the partition cannot learn it as a shortcut.
- **After any template change.** Run `make dataset`, then `make test`. Only new or changed requests need extraction, because the cache is keyed on prompt content.
- **Reviewing templates.** Check counts, duplicates and typos with scripts that print indices and counts, never the bad-request text.

## Conventions

- **Layers.** `hidden_states[l]` is the residual stream after block `l`, 1-based. A hook for layer `l` registers on `layers[l-1]`.
- **Cache.** Activations are cached per model, keyed on the full prompt content: system prompt, tools, user turn and assistant prefix. The item id is not part of the key. Old id-based cache files are renamed automatically on first use.
- **Configs.** Each config names its own `cache_dir` and `report_dir`. Runs for different models never overwrite each other. Make targets take `CONFIG=`, and `make preflight` takes `PREFLIGHT_CONFIG=`.
- **Splits.** The split is by family with `holdout_frac: 0.5`, so two families per policy are held out. S3 and S4 reuse the S2 split.
- **Layer choice.** The layer is chosen by leave-one-family-out on the training families. The test set is never used to pick it. Test AUROC for every layer is reported for reference only.
- **Thresholds.** Set on training negatives at `target_fpr` (1%), then applied unchanged to test and external data.
- **Confidence intervals.** S2 resamples whole requests, so wrapper variants of one request move together.
- **Sensitive files.** `s3_responses.jsonl` holds raw model responses, including harmful tool calls the model generated. It is git-ignored and must never be committed or shared.
- **External data.** Downloaded datasets go in their own folder under `data/external/`, which is git-ignored. Treat them as untrusted data, and check each license before use. XSTest is CC BY 4.0.
- **Commits.** No co-author line in commit messages. Push from the host, not the dev container.

## Open items

- O1. **Pending data.** `data/templates/incoming/cred_exfil.yaml` has a `secret_managers` family that `data/templates/cred_exfil.yaml` does not have. Those items are not in the dataset. Add the family to the template, merge, and rebuild.
- O2. **Thresholds do not transfer to new families.** On 0.5B, a 1% training false-alarm rate became 7% to 18% on benign execution requests from held-out families. The fix to test is calibration on a wider benign set, such as all benign requests plus an external set of benign commands like NL2Bash.
- O3. **The outbound gate is not usable.** Separating bad tool calls from benign ones is weaker on responses (0.81 to 0.96 AUROC) than on requests (0.96 to 1.00). The per-token and running-mean gates flag 17% to 50% of benign tool calls. Check whether 3B changes this before more work on it.
- O4. **Wrapper count.** 11 wrappers turn 484 distinct requests into 5,324 items. That roughly triples extraction time and adds little information. Reducing to about 3 is an option once 3B results are in.
- O5. **Small test sets.** Held-out sets are 26 to 34 distinct bad requests per policy inbound and 17 to 21 bad responses outbound, so intervals are wide. More distinct bad requests per family is the remedy, written under the Data workflow rules.
- O6. **Not tested yet.** No external benign execution set, no agent benchmark such as AgentHarm, no obfuscation or adaptive attack, and no baseline stronger than TF-IDF.

## Next on the GPU machine

1. Clone, switch the torch index in `pyproject.toml` to the CUDA build, run `uv lock` and `make setup`, and confirm CUDA is available. See the README.
2. `make preflight PREFLIGHT_CONFIG=configs/preflight_gpu.yaml`.
3. `make s2 s3 s4 CONFIG=configs/qwen3b.yaml`.
4. Fetch XSTest as the README shows, then `make xstest CONFIG=configs/qwen3b.yaml`.
5. Add the 3B results to `reports/FINDINGS.md` next to the 0.5B results, then decide on O2 to O4.

## Out of scope

vLLM integration, latency benchmarks, obfuscation and adaptive attack studies, second model family, product packaging. These were in the raw proposal and can be revisited after S5.

## Layout

```
kernelguard/data      schema, seeds, dataset builder, template merge command
data/templates        request templates per policy (yaml), plus incoming/ for files to merge
data/v1               built dataset (make dataset)
data/external         downloaded external datasets, git-ignored
kernelguard/extract   activation capture, content-keyed cache, batched GPU path
kernelguard/probes    probe bank (difference of means, logistic)
kernelguard/runtime   HF hook: inbound bank at prefill, outbound bank on generated tokens
kernelguard/eval      metrics, regex baseline
configs/              smoke_cpu, s2_cpu, preflight_cpu, preflight_gpu, qwen3b, llama3b
scripts/              smoke.py, s2_eval.py, s3_out.py, s4_text.py, external_xstest.py, make_preflight.py
reports/              findings note and results; reports/<model>/ for GPU configs
```
