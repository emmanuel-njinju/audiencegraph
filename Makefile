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

.PHONY: setup data identity segmentation lookalike propensity ctr campaign results modules test clean all

setup:            ## create venv + install pinned deps
	python3 -m venv .venv && $(PY) -m pip install -q --upgrade pip && $(PY) -m pip install -q -r requirements.txt

data:             ## generate the synthetic consumer universe (PEOPLE=n to scale)
	$(PY) data/generate_data.py --people $(PEOPLE) --out $(DATA) --seed 7

identity:         ## R1 - identity resolution engine (Cross-Device; graph)
	$(PY) src/identity/resolve.py --data $(DATA) --out reports/identity_metrics.json

segmentation:     ## R2 - audience segmentation (clustering)
	$(PY) src/segmentation/segment.py --data $(DATA) --out reports/segmentation_metrics.json

lookalike:        ## R2 - lookalike expansion (collaborative filtering)
	$(PY) src/lookalike/expand.py --data $(DATA) --out reports/lookalike_metrics.json

propensity:       ## R3 - behavioral propensity (classification)
	$(PY) src/propensity/model.py --data $(DATA) --out reports/propensity_metrics.json

ctr:              ## R3 - CTR/conversion optimizer (regression at scale)
	$(PY) src/ctr/model.py --data $(DATA) --out reports/ctr_metrics.json

campaign:         ## R3 - campaign optimizer (Bayesian A/B + bandit)
	$(PY) src/campaign/optimize.py --data $(DATA) --out reports/campaign_metrics.json

results:          ## build the results walkthrough (figure + RESULTS.md) from all metrics
	$(PY) scripts/build_results.py && $(PY) scripts/update_docs_metrics.py

modules: identity segmentation lookalike propensity ctr campaign results  ## run every module

test:             ## run the unit tests
	$(PY) -m pytest -q

all: data modules ## generate data then run every module + results

clean:            ## remove generated data + outputs (keep committed reports/)
	rm -rf data/synthetic outputs spark-warehouse metastore_db derby.log
