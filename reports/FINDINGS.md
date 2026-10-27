# KernelGuard-LLM: Findings on Qwen2.5-0.5B

Date: 2026-10-27. Model: Qwen2.5-0.5B-Instruct on CPU. Status: development result. The Qwen2.5-3B run is next.

## Question

Can a partition learned on an open-weight model's internal vectors classify requests on the way in and responses on the way out as good or bad, without parsing text?

## Answer on this model

On the way in, yes. The partition separates bad requests from benign execution requests on unseen request types at an AUROC of 0.96 to 1.00. At its operating threshold it beats a text classifier by a wide margin, and it never flagged a discussion question.

On the way out, only partly. Finished responses classify well. Separating a bad tool call from a benign one is weaker, and the per-token runtime gate raises too many false alarms to use.

## Method

- **Vectors.** One forward pass per request through the frozen model. The vector is the residual stream at the last prompt token for the inbound side, and the mean over generated tokens for the outbound side.
- **Partition.** Logistic regression on centered, unit-length vectors at one layer. The layer is chosen by leave-one-family-out over the training families only.
- **Threshold.** Set at 1% false alarms on training negatives, then applied unchanged to the test set.
- **Data.** 3 policies (shell execution, credential exfiltration, SQL mutation) with 4 request families each. There are 484 distinct requests, each phrased up to 11 ways, for 5,324 items. Good requests are benign execution requests plus questions about both harmful and benign topics.
- **Split.** Two whole families per policy are held out, so test requests differ in type and phrasing from training requests. Confidence intervals resample whole requests.
- **Outbound label.** A response is bad when the request was bad and the model made a tool call. Refusals and benign tool calls are good.
- **Baseline.** A TF-IDF logistic classifier on the request or response text, with the same splits and threshold rule. The better of word and character features is shown.

## Inbound results

Layer 11 of 24.

| Policy | Held-out bad requests | AUROC, bad vs benign execution | Bad caught | Benign execution flagged | Discussion flagged |
|---|---|---|---|---|---|
| CRED_EXFIL, vectors | 27 | 0.98 [0.96, 1.00] | 98% | 18% | 0% |
| CRED_EXFIL, text | 27 | 0.96 | 100% | 50% | 24% |
| SHELL_EXEC, vectors | 34 | 0.96 [0.92, 0.99] | 93% | 16% | 0% |
| SHELL_EXEC, text | 34 | 0.96 | 100% | 72% | 5% |
| SQL_MUTATION, vectors | 26 | 1.00 [1.00, 1.00] | 100% | 7% | 0% |
| SQL_MUTATION, text | 26 | 0.94 | 100% | 47% | 8% |

- **The intent signal generalises.** Ranking holds on request types the partition never saw. The lower confidence bounds stay at or above 0.92.
- **The advantage over text is at the operating point.** Both methods catch nearly all bad requests. At that threshold, text flags half or more of benign execution requests and the vectors flag 7% to 18%.
- **Topic alone does not trigger it.** No question about a harmful topic crossed the threshold.
- **Calibration does not carry over to new request types.** A 1% training false-alarm rate became 7% to 18% on the held-out families.
- **Middle layers work best.** Test AUROC peaks around layers 11 to 17 and falls toward the last layer. The 75%-depth rule from the original proposal gives layer 18, which is slightly worse.

## Outbound results

Same layer and family split, one phrasing per request.

| Policy | Bad responses in test | AUROC, all responses | AUROC, tool calls only, vectors | AUROC, tool calls only, text |
|---|---|---|---|---|
| CRED_EXFIL | 17 | 0.93 [0.85, 0.99] | 0.81 | 0.59 |
| SHELL_EXEC | 21 | 0.99 [0.96, 1.00] | 0.87 | 0.81 |
| SQL_MUTATION | 21 | 0.99 [0.98, 1.00] | 0.96 | 0.21 |

- **Vectors beat text at telling bad actions from benign ones.** On SQL the text classifier does worse than chance, while the vectors score 0.96.
- **The response vector carries intent less cleanly than the request vector.** Inbound, bad against benign execution scored 0.96 to 1.00. Outbound, bad against benign tool calls scored 0.81 to 0.96.

## Runtime gate

The gate scores the last prompt token with the inbound partition. It then scores each generated token, or the running mean of generated tokens, with the outbound partition.

- **Bad requests stop at prefill.** In the demo, all 3 held-out bad requests tripped at the last prompt token, before the model emitted any content.
- **The outbound part causes false alarms.** The running-mean gate flagged 17% to 42% of held-out benign tool calls. In the demo it stopped 2 of 3 benign tool calls about 18 tokens in. Scoring each token on its own was worse.
- **A usable gate today would be inbound only.** That gate would still need a threshold calibrated on a wider benign set.

## External check: XSTest

XSTest (Röttger et al., 2024) has 250 safe prompts that sound alarming, such as "kill" a process, and 200 unsafe contrast prompts. None of them were used in training. Each prompt was placed in the same agent transcript and scored with the S2 partition and thresholds, unchanged.

| Method | Safe prompts flagged | Unsafe prompts flagged |
|---|---|---|
| Vector partition, any policy | 0% | 0% |
| TF-IDF text classifier, any policy | 44% | 53% |

- **No false alarms on harmless prompts that sound alarming.** The text classifier, trained on the same requests, flags 44% of them.
- **The partition is specific to its policies, not to harm in general.** XSTest's unsafe prompts are about violence, discrimination and privacy, which are outside the three policies. The partition flags none of them, and its scores do not separate unsafe from safe (AUROC 0.50).
- **This test does not cover the known weak spot.** XSTest prompts are questions, and the partition already ignores questions. The open problem is false alarms on benign execution requests from new families (7% to 18%). An external set of benign commands, such as NL2Bash, would test that.

## Limits

- **Model.** The model is small and a weak agent. It made a tool call for 57% to 80% of bad requests but only 26% to 43% of benign execution requests.
- **Test size.** Held-out sets are 26 to 34 distinct bad requests inbound and 17 to 21 bad responses outbound, so the intervals are wide.
- **Data source.** All training requests come from templates written by one team. The only external test so far is XSTest.
- **Not tested.** Obfuscation and adaptive attacks were not run. An adaptive attacker is known to defeat activation probes (Bailey et al., ICLR 2026).
- **Weak baseline.** The text baseline is TF-IDF only. An embedding model or an LLM judge would be a stronger comparison.

## Next

1. Run on Qwen2.5-3B on the workstation GPU with `make s2 s3 s4 CONFIG=configs/qwen3b.yaml`.
2. Test false alarms on external benign execution requests, such as NL2Bash, and on agent requests from AgentHarm.
3. Calibrate the inbound threshold on a wider benign set and measure the false-alarm rate again.

## Reproduce

```
make dataset   # data/v1/items.jsonl from seeds and templates
make s2        # extract vectors (about 90 min on CPU) and run the smoke checks
make s2-eval   # inbound results, reports/s2.json
make s3        # outbound results and runtime demo, reports/s3.json (about 70 min on CPU)
make s4        # text baseline, reports/s4.json
make xstest    # external check, needs data/external/xstest/xstest_prompts.csv
```

`reports/s3_responses.jsonl` holds the model's raw responses, including harmful tool calls it generated. Do not share it.
