# AudienceGraph - developer entrypoints.
# Every target runs the same code that would run on EMR; only the Spark master
# and the data path change (see README "Running on AWS").

PY ?= .venv/bin/python
PEOPLE ?= 20000
DATA ?= data/synthetic
# Give the local driver JVM real heap (iterative graph joins + MLlib).
export AG_DRIVER_MEM ?= 4g
export PYSPARK_SUBMIT_ARGS ?= --driver-memory $(AG_DRIVER_MEM) pyspark-shell
export PYTHONPATH := .

.PHONY: setup data identity test clean all

setup:            ## create venv + install pinned deps
	python3 -m venv .venv && $(PY) -m pip install -q --upgrade pip && $(PY) -m pip install -q -r requirements.txt

data:             ## generate the synthetic consumer universe (PEOPLE=n to scale)
	$(PY) data/generate_data.py --people $(PEOPLE) --out $(DATA) --seed 7

identity:         ## Release 1 - run the identity resolution engine, write metrics
	$(PY) src/identity/resolve.py --data $(DATA) --out reports/identity_metrics.json

test:             ## run the unit tests
	$(PY) -m pytest -q

all: data identity ## generate data then run the identity engine

clean:            ## remove generated data + outputs (keep committed reports/)
	rm -rf data/synthetic outputs spark-warehouse metastore_db derby.log
