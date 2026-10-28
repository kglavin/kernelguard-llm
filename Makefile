CONFIG ?= configs/s2_cpu.yaml
PREFLIGHT_CONFIG ?= configs/preflight_cpu.yaml
LIMIT ?= 0

.PHONY: setup seeds dataset merge test smoke s2 s2-eval s3 s4 preflight xstest sessions sessions-eval clean

setup:
	uv sync --extra dev --python 3.12

seeds:
	uv run python -m kernelguard.data.make_seeds

dataset: seeds
	uv run python -m kernelguard.data.make_dataset

merge:
	uv run python -m kernelguard.data.merge_templates $(FILE)

test:
	uv run pytest -q

s2: dataset
	uv run python scripts/smoke.py $(CONFIG)

s2-eval:
	uv run python scripts/s2_eval.py $(CONFIG)

s3: s2-eval
	uv run python scripts/s3_out.py $(CONFIG)

s4:
	uv run python scripts/s4_text.py $(CONFIG)

preflight: dataset
	uv run python scripts/make_preflight.py data/v1/items.jsonl reports/preflight/items.jsonl
	uv run python scripts/smoke.py $(PREFLIGHT_CONFIG)
	uv run python scripts/s2_eval.py $(PREFLIGHT_CONFIG)
	uv run python scripts/s3_out.py $(PREFLIGHT_CONFIG) --max-new-tokens 24
	uv run python scripts/s4_text.py $(PREFLIGHT_CONFIG)

xstest:
	uv run python scripts/external_xstest.py $(CONFIG)

# private, git-ignored data from local Claude Code sessions, evaluation and calibration only
sessions:
	uv run python -m kernelguard.data.import_sessions

sessions-eval:
	uv run python scripts/private_sessions.py $(CONFIG) --limit $(LIMIT)

smoke: seeds
	uv run python scripts/smoke.py configs/smoke_cpu.yaml

clean:
	rm -rf reports/cache reports/smoke.json
