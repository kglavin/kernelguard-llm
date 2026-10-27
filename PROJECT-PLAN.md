# KernelGuard-LLM: Plan

Source: [raw-proposal.md](raw-proposal.md). Status: v3, 2026-10-27. S0 to S5 done on Qwen2.5-0.5B, see [reports/FINDINGS.md](reports/FINDINGS.md). Qwen2.5-3B run next.

## Thesis

Tap one layer of an open-weight LLM on the way in (prompt) and on the way out (response). Learn a partition of that vector space into good and bad from labeled examples, using classical ML on the vectors. Show the partition classifies held-out requests and responses without parsing text.

## Roles

- The LLM is a frozen feature extractor. One forward pass per example yields one vector at layer L.
- The partition is a scikit-learn model on those vectors. Training takes seconds. Inference is a dot product applied inside the forward pass.

## Steps

- S0 Scaffold (done). Schema, extraction, probe bank, runtime hook, metrics, CPU smoke on Qwen2.5-0.5B. `make smoke`.
- S1 Data (done). Enumerated requests labeled good or bad, generated from templates with slot fills, split by template family. Bad = a request to execute something harmful. Good = benign execution requests and prose discussion of the harmful thing. Templates live in `data/templates/*.yaml`. The benign and discussion templates are filled in. The malicious templates are seeded from the proposal's threat matrix and are extended by hand. `make dataset`.
- S2 In (done, 0.5B). Vector at the last prompt token at layer L. Fit difference-of-means, logistic regression, and k-NN. Report held-out accuracy and AUROC per policy. Sweep L.
- S3 Out (done, 0.5B). Let the model answer each request. Pool the vectors over the generated tokens at layer L. Label each response good or bad by whether it complied. Fit the same partitions, report the same numbers.
- S4 Comparison (done, 0.5B). One TF-IDF logistic classifier on the request text, same splits, so "vector space, not text" has a number beside it.
- S5 Result (0.5B note done, 3B pending). One script, one table, one page of findings in `reports/`. CPU 0.5B first, then Qwen2.5-3B on the workstation.

## Fixed choices

- Model: Qwen2.5-0.5B-Instruct on CPU for development, Qwen2.5-3B-Instruct on the workstation GPU for the result.
- Bad means a malicious user request. Prompt injection via documents is out of scope.
- Policies: shell exec, credential exfil, SQL mutation.
- Output: internal findings note, no paper.

## Success

S2 or S3 reaching held-out AUROC at or above 0.95 on at least two policies confirms the thesis for this model. Lower numbers are still the result. Either way S5 ships.

## Out of scope

vLLM integration, latency benchmarks, obfuscation and adaptive attack studies, second model family, product packaging. These were in the raw proposal and can be revisited after S5.

## Layout

```
kernelguard/data      schema, seeds, template generator
data/templates        request templates per policy (yaml), edited by hand
kernelguard/extract   activation capture, per-layer cache
kernelguard/probes    partitions (dom, logistic, knn)
kernelguard/runtime   HF hook that applies a partition during generation
kernelguard/eval      metrics, text baseline
configs/              smoke_cpu.yaml, qwen3b.yaml
scripts/              smoke.py, and one script per step
reports/              outputs
```
