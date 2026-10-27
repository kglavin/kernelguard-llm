CONFIG ?= configs/s2_cpu.yaml

.PHONY: setup seeds dataset merge test smoke s2 s2-eval s3 s4 clean

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

smoke: seeds
	uv run python scripts/smoke.py configs/smoke_cpu.yaml

clean:
	rm -rf reports/cache reports/smoke.json
