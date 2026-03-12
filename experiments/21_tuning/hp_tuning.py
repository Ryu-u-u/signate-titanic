"""Hyperparameter Tuning with Optuna + Weighted Voting Optimization.

Phase 1: Optuna で各モデルのハイパーパラメータを最適化
Phase 2: チューニング済みモデルで Weighted Voting の重みを最適化
"""
import sys
sys.path.append("../..")

import numpy as np
import pandas as pd
import yaml
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

from src.config import TRAIN_CSV, TEST_CSV, SAMPLE_SUBMIT_CSV, SEED, N_FOLDS, TARGET_COL
from src.utils import seed_everything
from src.features import compute_train_stats, build_pipeline
from src.evaluation import cross_validate

seed_everything(SEED)

# ============================================================
# Data Loading
# ============================================================
train = pd.read_csv(TRAIN_CSV, index_col=0)
test = pd.read_csv(TEST_CSV, index_col=0)

y = train[TARGET_COL]
X_raw = train.drop(columns=[TARGET_COL])

train_stats = compute_train_stats(X_raw)
X_train = build_pipeline(X_raw, version="v1", train_stats=train_stats)
X_test = build_pipeline(test, version="v1", train_stats=train_stats)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ============================================================
# Phase 1: Optuna Hyperparameter Tuning
# ============================================================
N_TRIALS = 100

def evaluate_model(model_fn):
    """CV AUCを返す"""
    _, mean_m = cross_validate(model_fn, X_train, y)
    return mean_m["auc"]


# --- LightGBM ---
def objective_lgbm(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": SEED,
        "verbose": -1,
    }
    return evaluate_model(lambda: LGBMClassifier(**params))


# --- XGBoost ---
def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": SEED,
        "eval_metric": "logloss",
    }
    return evaluate_model(lambda: XGBClassifier(**params))


# --- Random Forest ---
def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7]),
        "random_state": SEED,
    }
    return evaluate_model(lambda: RandomForestClassifier(**params))


# --- Logistic Regression ---
def objective_logreg(trial):
    C = trial.suggest_float("C", 0.001, 100.0, log=True)
    solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])
    params = {
        "C": C,
        "solver": solver,
        "max_iter": 2000,
        "random_state": SEED,
    }
    return evaluate_model(
        lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**params)),
        ])
    )


print("=" * 60)
print("Phase 1: Hyperparameter Tuning with Optuna")
print("=" * 60)

best_params = {}

for name, objective in [
    ("LightGBM", objective_lgbm),
    ("XGBoost", objective_xgb),
    ("RandomForest", objective_rf),
    ("LogReg", objective_logreg),
]:
    print(f"\n--- {name} ({N_TRIALS} trials) ---")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    best_params[name] = study.best_params
    print(f"  Best AUC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")


# ============================================================
# Phase 1.5: Verify tuned models
# ============================================================
print("\n" + "=" * 60)
print("Tuned Model Verification")
print("=" * 60)

def make_tuned_logreg():
    p = best_params["LogReg"]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=p["C"], solver=p["solver"],
            max_iter=2000, random_state=SEED,
        )),
    ])

def make_tuned_rf():
    p = best_params["RandomForest"]
    return RandomForestClassifier(**p, random_state=SEED)

def make_tuned_xgb():
    p = best_params["XGBoost"]
    return XGBClassifier(**p, random_state=SEED, eval_metric="logloss")

def make_tuned_lgbm():
    p = best_params["LightGBM"]
    return LGBMClassifier(**p, random_state=SEED, verbose=-1)

tuned_results = {}
for name, fn in [
    ("LogReg", make_tuned_logreg),
    ("RF", make_tuned_rf),
    ("XGB", make_tuned_xgb),
    ("LGBM", make_tuned_lgbm),
]:
    _, mean_m = cross_validate(fn, X_train, y)
    tuned_results[name] = mean_m
    print(f"  {name}: AUC={mean_m['auc']:.4f}, Acc={mean_m['accuracy']:.4f}")


# ============================================================
# Phase 2: Weighted Voting Optimization
# ============================================================
print("\n" + "=" * 60)
print("Phase 2: Weighted Voting Optimization")
print("=" * 60)

# Equal-weight Soft Voting (baseline)
def make_equal_voting():
    return VotingClassifier(
        estimators=[
            ("logreg", make_tuned_logreg()),
            ("rf", make_tuned_rf()),
            ("xgb", make_tuned_xgb()),
            ("lgbm", make_tuned_lgbm()),
        ],
        voting="soft",
    )

_, eq_voting_m = cross_validate(make_equal_voting, X_train, y)
print(f"  Equal Voting (tuned): AUC={eq_voting_m['auc']:.4f}")


# Grid search for weights
def objective_weights(trial):
    w_lr = trial.suggest_float("w_logreg", 0.0, 2.0)
    w_rf = trial.suggest_float("w_rf", 0.0, 2.0)
    w_xgb = trial.suggest_float("w_xgb", 0.0, 2.0)
    w_lgbm = trial.suggest_float("w_lgbm", 0.0, 2.0)
    weights = [w_lr, w_rf, w_xgb, w_lgbm]

    def make_weighted():
        return VotingClassifier(
            estimators=[
                ("logreg", make_tuned_logreg()),
                ("rf", make_tuned_rf()),
                ("xgb", make_tuned_xgb()),
                ("lgbm", make_tuned_lgbm()),
            ],
            voting="soft",
            weights=weights,
        )
    return evaluate_model(make_weighted)


print("\n  Optimizing weights (50 trials)...")
weight_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
)
weight_study.optimize(objective_weights, n_trials=50, show_progress_bar=True)

best_weights = weight_study.best_params
print(f"  Best weights: LR={best_weights['w_logreg']:.3f}, "
      f"RF={best_weights['w_rf']:.3f}, "
      f"XGB={best_weights['w_xgb']:.3f}, "
      f"LGBM={best_weights['w_lgbm']:.3f}")
print(f"  Weighted Voting AUC: {weight_study.best_value:.4f}")


# ============================================================
# Final Comparison
# ============================================================
print("\n" + "=" * 60)
print("Final Comparison")
print("=" * 60)

baseline = {
    "LogReg": 0.8536,
    "RF": 0.8603,
    "XGB": 0.8561,
    "LGBM": 0.8429,
    "Voting(equal)": 0.8639,
}

print(f"\n{'Model':<25} {'Baseline AUC':>14} {'Tuned AUC':>12} {'Diff':>8}")
print("-" * 60)
for name in ["LogReg", "RF", "XGB", "LGBM"]:
    base = baseline[name]
    tuned = tuned_results[name]["auc"]
    print(f"  {name:<23} {base:>12.4f} {tuned:>12.4f} {tuned - base:>+8.4f}")

base_voting = baseline["Voting(equal)"]
tuned_eq = eq_voting_m["auc"]
print(f"  {'Voting(equal,tuned)':<23} {base_voting:>12.4f} {tuned_eq:>12.4f} {tuned_eq - base_voting:>+8.4f}")
print(f"  {'Voting(weighted,tuned)':<23} {'-':>12} {weight_study.best_value:>12.4f} {weight_study.best_value - base_voting:>+8.4f}")


# ============================================================
# Generate Submissions
# ============================================================
print("\n" + "=" * 60)
print("Generating Submissions")
print("=" * 60)

sample_submit = pd.read_csv(SAMPLE_SUBMIT_CSV, index_col=0, header=None)

# 1. Equal-weight Voting (tuned models)
model_eq = make_equal_voting()
model_eq.fit(X_train, y)
pred_eq = model_eq.predict_proba(X_test)[:, 1]
sub_eq = sample_submit.copy()
sub_eq[1] = pred_eq
sub_eq.to_csv("submit_voting_tuned.csv", header=None)
print(f"  submit_voting_tuned.csv: mean={pred_eq.mean():.3f}, std={pred_eq.std():.3f}")

# 2. Weighted Voting (tuned models)
w = [best_weights["w_logreg"], best_weights["w_rf"],
     best_weights["w_xgb"], best_weights["w_lgbm"]]
model_wv = VotingClassifier(
    estimators=[
        ("logreg", make_tuned_logreg()),
        ("rf", make_tuned_rf()),
        ("xgb", make_tuned_xgb()),
        ("lgbm", make_tuned_lgbm()),
    ],
    voting="soft",
    weights=w,
)
model_wv.fit(X_train, y)
pred_wv = model_wv.predict_proba(X_test)[:, 1]
sub_wv = sample_submit.copy()
sub_wv[1] = pred_wv
sub_wv.to_csv("submit_voting_weighted.csv", header=None)
print(f"  submit_voting_weighted.csv: mean={pred_wv.mean():.3f}, std={pred_wv.std():.3f}")

# 3. Best single model (submit too for comparison)
best_single_name = max(tuned_results, key=lambda k: tuned_results[k]["auc"])
best_single_auc = tuned_results[best_single_name]["auc"]
print(f"\n  Best single model: {best_single_name} (AUC={best_single_auc:.4f})")

best_model_map = {
    "LogReg": make_tuned_logreg,
    "RF": make_tuned_rf,
    "XGB": make_tuned_xgb,
    "LGBM": make_tuned_lgbm,
}
best_single = best_model_map[best_single_name]()
best_single.fit(X_train, y)
pred_bs = best_single.predict_proba(X_test)[:, 1]
sub_bs = sample_submit.copy()
sub_bs[1] = pred_bs
sub_bs.to_csv(f"submit_{best_single_name.lower()}_tuned.csv", header=None)
print(f"  submit_{best_single_name.lower()}_tuned.csv: mean={pred_bs.mean():.3f}, std={pred_bs.std():.3f}")

print("\nDone!")
