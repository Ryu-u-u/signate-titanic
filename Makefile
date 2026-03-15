.PHONY: setup run-tuning run-feature-review run-all clean help \
	run-exp23 run-exp24 run-exp25 run-exp26 run-exp27 run-exp28 \
	run-exp29 run-exp30 run-exp31 run-exp32 run-exp33 run-exp34 \
	run-exp35 run-exp36 \
	run-priority-experiments run-blend-strategy evaluate \
	docs-build docs-serve

# Default target
help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-25s %s\n", $$1, $$2}'

setup: ## Set up the environment with uv
	uv sync

run-tuning: ## Run hyperparameter tuning (experiments/21_tuning)
	uv run python experiments/21_tuning/hp_tuning.py

run-feature-review: ## Run feature review (experiments/22_feature_review)
	uv run python experiments/22_feature_review/feature_review.py
	uv run python experiments/22_feature_review/distribution_analysis.py

run-exp23: ## Run HP re-tuning for domain+missing features
	cd experiments/23_hp_retune_domain_missing && uv run python hp_retune.py

run-exp24: ## Run CatBoost + 5-model Voting
	cd experiments/24_catboost && uv run python catboost_experiment.py

run-exp25: ## Run Advanced Stacking (nested CV)
	cd experiments/25_advanced_stacking && uv run python stacking.py

run-exp26: ## Run Multi-Seed Averaging
	cd experiments/26_multi_seed && uv run python multi_seed.py

run-exp27: ## Run Repeated-CV Robust Tuning
	cd experiments/27_repeated_cv && uv run python repeated_cv.py

run-exp28: ## Run Rank/Copula Ensemble
	cd experiments/28_rank_ensemble && uv run python rank_ensemble.py

run-exp29: ## Run CV-safe Target Encoding
	cd experiments/29_target_encoding && uv run python target_encoding.py

run-exp30: ## Run Calibration-First Ensembling
	cd experiments/30_calibration && uv run python calibration.py

run-exp31: ## Run Stability Feature Selection
	cd experiments/31_stability_selection && uv run python stability_selection.py

run-exp32: ## Run Agreement-Gated Pseudo Labeling
	cd experiments/32_pseudo_labeling && uv run python pseudo_labeling.py

run-exp33: ## Run Tabular Augmentation (Mixup)
	cd experiments/33_augmentation && uv run python augmentation.py

run-exp34: ## Run Bayesian Model Averaging
	cd experiments/34_bayesian && uv run python bayesian.py

run-exp35: ## Run Probability Blend Optimization (Strategies 1+2)
	cd experiments/35_probability_blend && uv run python blend.py

run-exp36: ## Run Hard Case Analysis + Feature Engineering (Strategy 3)
	cd experiments/36_hard_case_analysis && uv run python analysis.py

run-exp36b: ## Run Enriched Features with External Data (Strategy 3+)
	cd experiments/36_hard_case_analysis && uv run python enriched_features.py

run-submission-blend: ## Run Cross-Experiment Submission Blend
	cd experiments/35_probability_blend && uv run python submission_blend.py

run-blend-strategy: ## Run full blend strategy pipeline (exp35 -> exp36 -> exp36b -> blend)
	$(MAKE) run-exp35
	$(MAKE) run-exp36
	$(MAKE) run-exp36b
	$(MAKE) run-submission-blend

run-priority-experiments: ## Run top priority experiments (23, 26, 28, 27)
	$(MAKE) run-exp23
	$(MAKE) run-exp26
	$(MAKE) run-exp28
	$(MAKE) run-exp27

run-all: ## Run all experiment scripts in order
	uv run python experiments/21_tuning/hp_tuning.py
	uv run python experiments/22_feature_review/feature_review.py
	uv run python experiments/22_feature_review/distribution_analysis.py

evaluate: ## Evaluate a submission CSV locally (usage: make evaluate FILE=submit.csv)
	uv run python scripts/evaluate_submission.py $(FILE)

docs-build: ## Build docs site (generates docs/ from READMEs then runs mkdocs build)
	uv run python scripts/build_docs.py
	uv run mkdocs build --strict

docs-serve: ## Serve docs locally with hot reload (access from phone via LAN IP:8000)
	uv run python scripts/build_docs.py
	uv run mkdocs serve --dev-addr 0.0.0.0:8000

clean: ## Remove .venv and cache files
	rm -rf .venv __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
