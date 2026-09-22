# Convenience targets for the reproduction experiments.
PYTHON ?= python
DATA ?= data
RUNS ?= runs

.PHONY: install test data table2 cot tables plug-ablation scale vram all clean

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests -q

data:
	$(PYTHON) scripts/prepare_data.py --data-dir $(DATA)

cot:
	$(PYTHON) scripts/run_cot_baseline.py --config configs/strategyqa.yaml \
		--data-dir $(DATA) --output-dir $(RUNS)/baselines/strategyqa

# Table 2 (all datasets, settings and adapter sizes)
table2:
	$(PYTHON) reproduce_main_results.py --data-dir $(DATA) --output-root $(RUNS)/table2

# Tables 5 (ablation) and Figure 3 (beam / iteration scaling) on StrategyQA
ablations:
	$(PYTHON) scripts/ablation_loss.py --config configs/strategyqa.yaml \
		--data-dir $(DATA) --output-dir $(RUNS)/ablation/strategyqa
	$(PYTHON) scripts/scale_analysis.py --config configs/strategyqa.yaml \
		--run-dir $(RUNS)/table2/strategyqa/combined --data-dir $(DATA) \
		--output-dir $(RUNS)/scale/strategyqa

# Table 6 (requires a GPU)
vram:
	$(PYTHON) scripts/measure_vram.py --config configs/mixtral_strategyqa.yaml \
		--output-dir $(RUNS)/vram

tables:
	$(PYTHON) scripts/make_tables.py --runs-root $(RUNS)/table2 \
		--output-dir $(RUNS)/tables

# Everything, including the GPU-only stages
all:
	$(PYTHON) reproduce_all.py --data-dir $(DATA) --runs-root $(RUNS)

clean:
	rm -rf $(RUNS) __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
