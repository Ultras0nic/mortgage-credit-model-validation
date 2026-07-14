.PHONY: setup run test clean

setup:
	python -m pip install -e ".[dev]"

run:
	python -m mortgage_validation.pipeline

test:
	python -m pytest -q

clean:
	python -c "from pathlib import Path; import shutil; [shutil.rmtree(Path(p), ignore_errors=True) for p in ['data/generated','artifacts/model','artifacts/tables','artifacts/charts']]"

