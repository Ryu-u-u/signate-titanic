"""Experiment 24: CatBoost + 5-model Voting.

Hypothesis: CatBoost adds diversity to the ensemble via Ordered Boosting.

Phase 1: CatBoost baseline with default params + domain+missing features
Phase 2: Optuna HP tuning for CatBoost (100 trials)
Phase 3: 5-model Equal Voting (LogReg + RF + XGB + LGBM + CatBoost)
Phase 4: Generate submissions (5-model voting, best CatBoost single)
"""
import sys
sys.path.append("../..")

import numpy as np
import pandas as pd
import optuna
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.config import (
    TRAIN_CSV, TEST_CSV, SAMPLE_SUBMIT_CSV, SEED, N_FOLDS, TARGET_COL,
)
from src.utils import seed_everything
from src.features import compute_train_stats, build_pipeline
from src.evaluation import cross_validate
from src.exp_features import make_exp_builder

seed_everything(SEED)

# ============================================================
# Constants
# ============================================================
N_TRIALS = 100

BEST_PARAMS = {
    "LightGBM": {
        "n_estimators": 623, "learning_rate": 0.013317600672042055,
        "num_leaves": 18, "min_child_samples": 38,
        "subsample": 0.6580623204210281, "colsample_bytree": 0.8110201909037074,
        "reg_alpha": 0.05947005529609853, "reg_lambda": 9.841972911125754e-07,
    },
    "XGBoost": {
        "n_estimators": 635, "learning_rate": 0.01256683572112762,
        "max_depth": 5, "subsample": 0.9941111395416468,
        "colsample_bytree": 0.9393171998430397,
        "reg_lambda": 2.829234072503869e-06, "reg_alpha": 3.1211260160721054e-06,
        "min_child_weight": 6,
    },
    "RandomForest": {
        "n_estimators": 785, "max_depth": 6, "min_samples_leaf": 10,
        "min_samples_split": 17, "max_features": 0.7,
    },
    "LogReg": {
        "C": 0.04338580400456848, "solver": "liblinear",
    },
}

# ============================================================
# Data Loading
# ============================================================
print("=" * 60)
print("Experiment 24: CatBoost + 5-model Voting")
print("=" * 60)

train = pd.read_csv(TRAIN_CSV, index_col=0)
test = pd.read_csv(TEST_CSV, index_col=0)

y = train[TARGET_COL]
X_raw = train.drop(columns=[TARGET_COL])

print(f"Train: {X_raw.shape}, Test: {test.shape}")

# Feature builder: domain + missing
fb = make_exp_builder(
    missing_flags=True, age_bins=None, fare_bins=None,
    interactions=False, polynomial=False, group_stats=False,
    freq_encoding=False, rank_features=False, domain_features=True,
)

# ============================================================
# Model Factories (existing 4 models)
# ============================================================
def make_tuned_logreg():
    p = BEST_PARAMS["LogReg"]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=p["C"], solver=p["solver"],
                                     max_iter=2000, random_state=SEED)),
    ])

def make_tuned_rf():
    p = BEST_PARAMS["RandomForest"]
    return RandomForestClassifier(**p, random_state=SEED)

def make_tuned_xgb():
    p = BEST_PARAMS["XGBoost"]
    return XGBClassifier(**p, random_state=SEED, eval_metric="logloss", device="cuda")

def make_tuned_lgbm():
    p = BEST_PARAMS["LightGBM"]
    return LGBMClassifier(**p, random_state=SEED, verbose=-1)

def make_tuned_voting():
    return VotingClassifier(
        estimators=[
            ("logreg", make_tuned_logreg()),
            ("rf", make_tuned_rf()),
            ("xgb", make_tuned_xgb()),
            ("lgbm", make_tuned_lgbm()),
        ],
        voting="soft",
    )

# ============================================================
# Phase 1: CatBoost Baseline
# ============================================================
print("\n" + "=" * 60)
print("Phase 1: CatBoost Baseline (default params)")
print("=" * 60)

def make_catboost_default():
    return CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        random_seed=SEED, verbose=0,
        eval_metric="AUC",
    )

fold_metrics, mean_metrics = cross_validate(
    make_catboost_default, X_raw, y, feature_builder=fb,
)

print(f"  CatBoost (default):")
print(f"    AUC:      {mean_metrics['auc']:.6f}")
print(f"    Accuracy: {mean_metrics['accuracy']:.6f}")
print(f"    F1:       {mean_metrics['f1']:.6f}")
print(f"    LogLoss:  {mean_metrics['logloss']:.6f}")

for fm in fold_metrics:
    print(f"    Fold {fm['fold']}: AUC={fm['auc']:.6f}")

catboost_default_auc = mean_metrics["auc"]

# ============================================================
# Phase 2: Optuna HP Tuning for CatBoost
# ============================================================
print("\n" + "=" * 60)
print(f"Phase 2: Optuna CatBoost Tuning ({N_TRIALS} trials)")
print("=" * 60)

def objective_catboost(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        "bootstrap_type": "Bernoulli",
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_seed": SEED,
        "verbose": 0,
        "eval_metric": "AUC",
    }

    def model_fn():
        return CatBoostClassifier(**params)

    _, mean_m = cross_validate(model_fn, X_raw, y, feature_builder=fb)
    return mean_m["auc"]

study_catboost = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(),
    sampler=optuna.samplers.TPESampler(seed=SEED),
)
study_catboost.optimize(
    objective_catboost, n_trials=N_TRIALS, show_progress_bar=True,
)

catboost_best_params = study_catboost.best_params
print(f"  Best AUC: {study_catboost.best_value:.6f}")
print(f"  Default AUC: {catboost_default_auc:.6f}")
print(f"  Improvement: {study_catboost.best_value - catboost_default_auc:+.6f}")
print(f"  Best params: {catboost_best_params}")

# Build tuned CatBoost factory
def make_tuned_catboost():
    p = catboost_best_params
    return CatBoostClassifier(
        iterations=p["iterations"],
        learning_rate=p["learning_rate"],
        depth=p["depth"],
        l2_leaf_reg=p["l2_leaf_reg"],
        min_data_in_leaf=p["min_data_in_leaf"],
        bootstrap_type="Bernoulli",
        subsample=p["subsample"],
        random_seed=SEED,
        verbose=0,
        eval_metric="AUC",
    )

# Verify tuned CatBoost
fold_metrics_tuned, mean_metrics_tuned = cross_validate(
    make_tuned_catboost, X_raw, y, feature_builder=fb,
)
print(f"\n  Tuned CatBoost verification:")
print(f"    AUC:      {mean_metrics_tuned['auc']:.6f}")
print(f"    Accuracy: {mean_metrics_tuned['accuracy']:.6f}")
print(f"    F1:       {mean_metrics_tuned['f1']:.6f}")

for fm in fold_metrics_tuned:
    print(f"    Fold {fm['fold']}: AUC={fm['auc']:.6f}")

# ============================================================
# Phase 3: 5-model vs 4-model Voting
# ============================================================
print("\n" + "=" * 60)
print("Phase 3: 4-model vs 5-model Voting Comparison")
print("=" * 60)

def make_5model_voting():
    return VotingClassifier(
        estimators=[
            ("logreg", make_tuned_logreg()),
            ("rf", make_tuned_rf()),
            ("xgb", make_tuned_xgb()),
            ("lgbm", make_tuned_lgbm()),
            ("catboost", make_tuned_catboost()),
        ],
        voting="soft",
    )

# 4-model voting
_, mean_4model = cross_validate(
    make_tuned_voting, X_raw, y, feature_builder=fb,
)
print(f"\n  4-model Voting (LogReg+RF+XGB+LGBM):")
print(f"    AUC:      {mean_4model['auc']:.6f}")
print(f"    Accuracy: {mean_4model['accuracy']:.6f}")
print(f"    F1:       {mean_4model['f1']:.6f}")

# 5-model voting
fold_metrics_5m, mean_5model = cross_validate(
    make_5model_voting, X_raw, y, feature_builder=fb,
)
print(f"\n  5-model Voting (LogReg+RF+XGB+LGBM+CatBoost):")
print(f"    AUC:      {mean_5model['auc']:.6f}")
print(f"    Accuracy: {mean_5model['accuracy']:.6f}")
print(f"    F1:       {mean_5model['f1']:.6f}")

for fm in fold_metrics_5m:
    print(f"    Fold {fm['fold']}: AUC={fm['auc']:.6f}")

delta_voting = mean_5model["auc"] - mean_4model["auc"]
print(f"\n  Voting AUC change: {delta_voting:+.6f}")
if delta_voting > 0:
    print("  -> 5-model Voting is better!")
else:
    print("  -> 4-model Voting remains better (or equal).")

# All models comparison
print("\n  All models summary:")
all_models = {
    "LogReg": make_tuned_logreg,
    "RandomForest": make_tuned_rf,
    "XGBoost": make_tuned_xgb,
    "LightGBM": make_tuned_lgbm,
    "CatBoost (tuned)": make_tuned_catboost,
    "4-model Voting": make_tuned_voting,
    "5-model Voting": make_5model_voting,
}

summary = {}
for name, model_fn in all_models.items():
    _, mm = cross_validate(model_fn, X_raw, y, feature_builder=fb)
    summary[name] = mm
    print(f"    {name:20s}: AUC={mm['auc']:.6f}  Acc={mm['accuracy']:.6f}")

# ============================================================
# Phase 4: Generate Submissions
# ============================================================
print("\n" + "=" * 60)
print("Phase 4: Submission Generation")
print("=" * 60)

train_stats = compute_train_stats(X_raw)
X_full, X_test_feat = fb(X_raw, test)
sample = pd.read_csv(SAMPLE_SUBMIT_CSV, index_col=0, header=None)

# Submission 1: 5-model Voting
model_5m = make_5model_voting()
model_5m.fit(X_full, y)
proba_5m = model_5m.predict_proba(X_test_feat)[:, 1]

sub_5m = pd.DataFrame({0: sample.index, 1: proba_5m})
sub_5m.to_csv(
    "submission_5model_voting.csv",
    index=False, header=False,
)
print("  Saved: submission_5model_voting.csv")

# Submission 2: CatBoost single (tuned)
model_cat = make_tuned_catboost()
model_cat.fit(X_full, y)
proba_cat = model_cat.predict_proba(X_test_feat)[:, 1]

sub_cat = pd.DataFrame({0: sample.index, 1: proba_cat})
sub_cat.to_csv(
    "submission_catboost_tuned.csv",
    index=False, header=False,
)
print("  Saved: submission_catboost_tuned.csv")

# Print tuned CatBoost params for reference
print("\n--- Tuned CatBoost Params ---")
for k, v in catboost_best_params.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("Experiment 24 complete!")
print("=" * 60)
