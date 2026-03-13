.PHONY: setup run-tuning run-feature-review run-all clean help

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

run-all: ## Run all experiment scripts in order
	uv run python experiments/21_tuning/hp_tuning.py
	uv run python experiments/22_feature_review/feature_review.py
	uv run python experiments/22_feature_review/distribution_analysis.py

clean: ## Remove .venv and cache files
	rm -rf .venv __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
