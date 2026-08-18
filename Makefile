.PHONY: venv install pipeline pipeline-force lint test app clean smoke

PYTHON := python3
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Full local run: resumes from checkpoints in output/pipeline_state.json
pipeline:
	cd src && ../$(PY) run_pipeline.py

# Ignore checkpoints and rerun everything (use after changing model configs)
pipeline-force:
	cd src && ../$(PY) run_pipeline.py --force

# Fast smoke run using a tiny sample + 2 Optuna trials — mirrors CI
smoke:
	cd src && FRAUD_N_TRIALS=2 FRAUD_CV_FOLDS=2 FRAUD_DATA_PATH=$(CURDIR)/tests/fixtures/sample.csv ../$(PY) run_pipeline.py --force

lint:
	$(PY) -m flake8 src app tests --max-line-length=120

test:
	$(PY) -m pytest tests -v

app:
	$(VENV)/bin/streamlit run app/streamlit_app.py

clean:
	rm -rf output/* models/* .venv
