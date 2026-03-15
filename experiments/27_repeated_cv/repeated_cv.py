"""Experiment 27: Repeated-CV Robust Tuning.

Hypothesis: Tuning with `mean(AUC) - lambda*std(AUC)` rewards stability,
reducing the CV-to-Public gap.

Phase 1: Evaluate current BEST_PARAMS with repeated CV (5 seeds x 5 folds = 25 folds)
Phase 2: Optuna re-tuning with robust objective (50 trials per model)
Phase 3: Compare robust-tuned vs BEST_PARAMS (Equal Voting)
Phase 4: Generate submission if improved
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
CV_SEEDS = [42, 123, 456, 789, 2024]
LAMBDA = 0.5
N_TRIALS = 50  # reduced for repeated-CV (very expensive)

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
print("Experiment 27: Repeated-CV Robust Tuning")
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
# Model Factories (current BEST_PARAMS)
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
# Robust Evaluation Function
# ============================================================
def evaluate_robust(model_fn, feature_builder, cv_seeds=CV_SEEDS, lam=LAMBDA):
    """Evaluate model with repeated CV and return robust score.

    Returns
    -------
    robust_score : float
        mean(AUC) - lam * std(AUC)
    mean_auc : float
    std_auc : float
    """
    all_aucs = []
    for s in cv_seeds:
        fold_metrics, _ = cross_validate(
            model_fn, X_raw, y, seed=s, feature_builder=feature_builder,
        )
        all_aucs.extend([m["auc"] for m in fold_metrics])
    mean_auc = np.mean(all_aucs)
    std_auc = np.std(all_aucs)
    robust_score = mean_auc - lam * std_auc
    return robust_score, mean_auc, std_auc

# ============================================================
# Phase 1: Baseline Robust Evaluation
# ============================================================
print("\n" + "=" * 60)
print("Phase 1: Baseline Robust Evaluation (BEST_PARAMS)")
print(f"  CV seeds: {CV_SEEDS}, lambda: {LAMBDA}")
print("=" * 60)

baseline_results = {}

models_baseline = {
    "LogReg": make_tuned_logreg,
    "RandomForest": make_tuned_rf,
    "XGBoost": make_tuned_xgb,
    "LightGBM": make_tuned_lgbm,
    "Voting": make_tuned_voting,
}

for name, model_fn in models_baseline.items():
    robust, mean_auc, std_auc = evaluate_robust(model_fn, fb)
    baseline_results[name] = {
        "robust_score": robust,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
    }
    print(f"  {name:15s}: AUC={mean_auc:.6f} +/- {std_auc:.6f}  "
          f"robust={robust:.6f}")

# ============================================================
# Phase 2: Optuna Re-tuning with Robust Objective
# ============================================================
print("\n" + "=" * 60)
print("Phase 2: Optuna Robust Re-tuning")
print(f"  N_TRIALS={N_TRIALS}, 5x5 CV per trial")
print("=" * 60)

robust_best_params = {}

# --- LightGBM ---
print("\n--- LightGBM ---")

def objective_lgbm(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    model_fn = lambda: LGBMClassifier(**params, random_state=SEED, verbose=-1)
    robust_score, _, _ = evaluate_robust(model_fn, fb)
    return robust_score

study_lgbm = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(),
    sampler=optuna.samplers.TPESampler(seed=SEED),
)
study_lgbm.optimize(objective_lgbm, n_trials=N_TRIALS, show_progress_bar=True)
robust_best_params["LightGBM"] = study_lgbm.best_params
print(f"  Best robust score: {study_lgbm.best_value:.6f}")
print(f"  Best params: {study_lgbm.best_params}")

# --- XGBoost ---
print("\n--- XGBoost ---")

def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    }
    model_fn = lambda: XGBClassifier(**params, random_state=SEED, eval_metric="logloss", device="cuda")
    robust_score, _, _ = evaluate_robust(model_fn, fb)
    return robust_score

study_xgb = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(),
    sampler=optuna.samplers.TPESampler(seed=SEED),
)
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, show_progress_bar=True)
robust_best_params["XGBoost"] = study_xgb.best_params
print(f"  Best robust score: {study_xgb.best_value:.6f}")
print(f"  Best params: {study_xgb.best_params}")

# --- RandomForest ---
print("\n--- RandomForest ---")

def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "max_features": trial.suggest_categorical(
            "max_features", [0.3, 0.5, 0.7, "sqrt", "log2"],
        ),
    }
    model_fn = lambda: RandomForestClassifier(**params, random_state=SEED)
    robust_score, _, _ = evaluate_robust(model_fn, fb)
    return robust_score

study_rf = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(),
    sampler=optuna.samplers.TPESampler(seed=SEED),
)
study_rf.optimize(objective_rf, n_trials=N_TRIALS, show_progress_bar=True)
robust_best_params["RandomForest"] = study_rf.best_params
print(f"  Best robust score: {study_rf.best_value:.6f}")
print(f"  Best params: {study_rf.best_params}")

# --- LogReg ---
print("\n--- LogReg ---")

def objective_logreg(trial):
    params = {
        "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
        "solver": trial.suggest_categorical("solver", ["liblinear", "lbfgs"]),
    }
    model_fn = lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(**params, max_iter=2000, random_state=SEED)),
    ])
    robust_score, _, _ = evaluate_robust(model_fn, fb)
    return robust_score

study_logreg = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(),
    sampler=optuna.samplers.TPESampler(seed=SEED),
)
study_logreg.optimize(objective_logreg, n_trials=N_TRIALS, show_progress_bar=True)
robust_best_params["LogReg"] = study_logreg.best_params
print(f"  Best robust score: {study_logreg.best_value:.6f}")
print(f"  Best params: {study_logreg.best_params}")

# ============================================================
# Phase 3: Compare Robust-tuned vs BEST_PARAMS (Voting)
# ============================================================
print("\n" + "=" * 60)
print("Phase 3: Robust-tuned vs BEST_PARAMS Comparison")
print("=" * 60)

# Build robust-tuned model factories
def make_robust_logreg():
    p = robust_best_params["LogReg"]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=p["C"], solver=p["solver"],
            max_iter=2000, random_state=SEED,
        )),
    ])

def make_robust_rf():
    p = robust_best_params["RandomForest"]
    return RandomForestClassifier(**p, random_state=SEED)

def make_robust_xgb():
    p = robust_best_params["XGBoost"]
    return XGBClassifier(**p, random_state=SEED, eval_metric="logloss", device="cuda")

def make_robust_lgbm():
    p = robust_best_params["LightGBM"]
    return LGBMClassifier(**p, random_state=SEED, verbose=-1)

def make_robust_voting():
    return VotingClassifier(
        estimators=[
            ("logreg", make_robust_logreg()),
            ("rf", make_robust_rf()),
            ("xgb", make_robust_xgb()),
            ("lgbm", make_robust_lgbm()),
        ],
        voting="soft",
    )

# Evaluate robust-tuned individual models
robust_results = {}
models_robust = {
    "LogReg": make_robust_logreg,
    "RandomForest": make_robust_rf,
    "XGBoost": make_robust_xgb,
    "LightGBM": make_robust_lgbm,
    "Voting": make_robust_voting,
}

print("\nRobust-tuned model evaluation:")
for name, model_fn in models_robust.items():
    robust, mean_auc, std_auc = evaluate_robust(model_fn, fb)
    robust_results[name] = {
        "robust_score": robust,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
    }
    print(f"  {name:15s}: AUC={mean_auc:.6f} +/- {std_auc:.6f}  "
          f"robust={robust:.6f}")

# Comparison summary
print("\nComparison (BEST_PARAMS vs Robust-tuned):")
print(f"  {'Model':15s}  {'Baseline robust':>16s}  {'Robust-tuned':>16s}  {'Delta':>10s}")
print("-" * 65)
for name in ["LogReg", "RandomForest", "XGBoost", "LightGBM", "Voting"]:
    bl = baseline_results[name]["robust_score"]
    rb = robust_results[name]["robust_score"]
    delta = rb - bl
    marker = " *" if delta > 0 else ""
    print(f"  {name:15s}  {bl:16.6f}  {rb:16.6f}  {delta:+10.6f}{marker}")

# ============================================================
# Phase 4: Generate Submission (if improved)
# ============================================================
print("\n" + "=" * 60)
print("Phase 4: Submission Generation")
print("=" * 60)

bl_voting = baseline_results["Voting"]["robust_score"]
rb_voting = robust_results["Voting"]["robust_score"]

if rb_voting > bl_voting:
    best_voting_fn = make_robust_voting
    print("Robust-tuned Voting is better. Generating submission.")
    print(f"  Baseline robust={bl_voting:.6f} -> Robust-tuned={rb_voting:.6f}")
else:
    best_voting_fn = make_tuned_voting
    print("BEST_PARAMS Voting is better (or equal). Using baseline for submission.")
    print(f"  Baseline robust={bl_voting:.6f} -> Robust-tuned={rb_voting:.6f}")

# Train on full data and predict
train_stats = compute_train_stats(X_raw)
X_full, X_test_feat = fb(X_raw, test)

model = best_voting_fn()
model.fit(X_full, y)
proba = model.predict_proba(X_test_feat)[:, 1]

# Save submission
sample = pd.read_csv(SAMPLE_SUBMIT_CSV, index_col=0, header=None)
submission = pd.DataFrame({
    0: sample.index,
    1: proba,
})
submission.to_csv(
    "submission_robust.csv",
    index=False, header=False,
)
print("Submission saved: submission_robust.csv")

# Save robust params
print("\n--- Robust Best Params ---")
for name, params in robust_best_params.items():
    print(f"\n  {name}:")
    for k, v in params.items():
        print(f"    {k}: {v}")

print("\n" + "=" * 60)
print("Experiment 27 complete!")
print("=" * 60)
