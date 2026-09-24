PYTHON := .venv/bin/python
PIP := $(PYTHON) -m pip

.PHONY: help venv install test v1 v05 clean-reports

help:
	@echo "Available targets:"
	@echo "  make venv          Create the local virtual environment"
	@echo "  make install       Install project dependencies into .venv"
	@echo "  make test          Run the lightweight unit tests"
	@echo "  make v1            Run the V1 Gaussian notebook-faithful workflow"
	@echo "  make v05           Run the V0.5 Dirichlet notebook-faithful workflow"
	@echo "  make clean-reports Remove generated report folders"

venv:
	python3 -m venv .venv

install:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e '.[parquet,dev]'

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

v1:
	PYTHONPATH=src $(PYTHON) scripts/run_notebook_v1_gaussian.py

v05:
	PYTHONPATH=src $(PYTHON) scripts/run_notebook_v0_5_dirichlet.py

clean-reports:
	rm -rf reports/notebook_v1_gaussian reports/notebook_v0_5_dirichlet
