"""HP Re-Tuning for domain+missing Feature Set (~23 features).

BEST_PARAMS were tuned on v1 (14 features). This experiment re-tunes
hyperparameters for the domain+missing feature set to find optimal values
for the expanded feature space.

Phase 1: Optuna tuning for 4 models (LGBM, XGB, RF, LogReg) x 100 trials
Phase 2: Equal Voting (no weight optimization - overfits on Public scores)
Phase 3: Compare with previous BEST_PARAMS baseline
Phase 4: Generate submissions if improved
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

from src.config import TRAIN_CSV, TEST_CSV, SAMPLE_SUBMIT_CSV, SEED, N_FOLDS, TARGET_COL
from src.utils import seed_everything
from src.features import compute_train_stats, build_pipeline
from src.exp_features import make_exp_builder
from src.evaluation import cross_validate

seed_everything(SEED)

# ============================================================
# Data Loading
# ============================================================
train = pd.read_csv(TRAIN_CSV, index_col=0)
test = pd.read_csv(TEST_CSV, index_col=0)

y = train[TARGET_COL]
X_raw = train.drop(columns=[TARGET_COL])

print(f"X_raw: {X_raw.shape}")

# ============================================================
# Feature Builder: domain+missing (~23 features)
# ============================================================
fb = make_exp_builder(
    missing_flags=True, age_bins=None, fare_bins=None,
    interactions=False, polynomial=False, group_stats=False,
    freq_encoding=False, rank_features=False, domain_features=True,
)

# ============================================================
# Previous BEST_PARAMS (tuned on v1, 14 features)
# ============================================================
PREV_BEST_PARAMS = {
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
# Helper: evaluate with feature_builder
# ============================================================
def evaluate_model(model_fn):
    """Return CV AUC using domain+missing feature_builder on X_raw."""
    _, mean_m = cross_validate(model_fn, X_raw, y, feature_builder=fb)
    return mean_m["auc"]


# ============================================================
# Phase 1: Optuna Hyperparameter Tuning
# ============================================================
print("=" * 60)
print("Phase 1: Optuna HP Tuning for domain+missing (~23 features)")
print("=" * 60)

N_TRIALS = 100


# --- LightGBM ---
# Wider num_leaves (7-127) and colsample_bytree (0.3-1.0) for ~23 features
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
        "random_state": SEED,
        "verbose": -1,
    }
    return evaluate_model(lambda: LGBMClassifier(**params))


# --- XGBoost ---
# Wider max_depth (2-10) and colsample_bytree (0.3-1.0) for ~23 features
def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": SEED,
        "eval_metric": "logloss",
    }
    return evaluate_model(lambda: XGBClassifier(**params))


# --- Random Forest ---
# max_features options include 0.3 for wider search with more features
def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical(
            "max_features", [0.3, 0.5, 0.7, "sqrt", "log2"]
        ),
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


# Run tuning for all 4 models
best_params = {}

for name, objective in [
    ("LightGBM", objective_lgbm),
    ("XGBoost", objective_xgb),
    ("RandomForest", objective_rf),
    ("LogReg", objective_logreg),
]:
    print(f"\n--- {name} ({N_TRIALS} trials) ---")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    best_params[name] = study.best_params
    print(f"  Best AUC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")


# ============================================================
# Phase 1.5: Model factories with new best params
# ============================================================
def make_new_logreg():
    p = best_params["LogReg"]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=p["C"], solver=p["solver"],
            max_iter=2000, random_state=SEED,
        )),
    ])


def make_new_rf():
    p = best_params["RandomForest"]
    return RandomForestClassifier(**p, random_state=SEED)


def make_new_xgb():
    p = best_params["XGBoost"]
    return XGBClassifier(**p, random_state=SEED, eval_metric="logloss")


def make_new_lgbm():
    p = best_params["LightGBM"]
    return LGBMClassifier(**p, random_state=SEED, verbose=-1)


# ============================================================
# Phase 2: Equal Voting (NO weight optimization)
# ============================================================
print("\n" + "=" * 60)
print("Phase 2: Equal Voting with Re-Tuned Models")
print("=" * 60)


def make_new_voting():
    return VotingClassifier(
        estimators=[
            ("logreg", make_new_logreg()),
            ("rf", make_new_rf()),
            ("xgb", make_new_xgb()),
            ("lgbm", make_new_lgbm()),
        ],
        voting="soft",
    )


# Verify each re-tuned model individually
new_results = {}
for name, fn in [
    ("LogReg", make_new_logreg),
    ("RF", make_new_rf),
    ("XGB", make_new_xgb),
    ("LGBM", make_new_lgbm),
]:
    _, mean_m = cross_validate(fn, X_raw, y, feature_builder=fb)
    new_results[name] = mean_m
    print(f"  {name}: AUC={mean_m['auc']:.4f}, Acc={mean_m['accuracy']:.4f}")

_, new_voting_m = cross_validate(make_new_voting, X_raw, y, feature_builder=fb)
new_results["Voting"] = new_voting_m
print(f"  Voting(equal): AUC={new_voting_m['auc']:.4f}, Acc={new_voting_m['accuracy']:.4f}")


# ============================================================
# Phase 3: Compare with Previous BEST_PARAMS
# ============================================================
print("\n" + "=" * 60)
print("Phase 3: Comparison with Previous BEST_PARAMS (v1-tuned)")
print("=" * 60)


# Previous best models with domain+missing features
def make_prev_logreg():
    p = PREV_BEST_PARAMS["LogReg"]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=p["C"], solver=p["solver"],
            max_iter=2000, random_state=SEED,
        )),
    ])


def make_prev_rf():
    p = PREV_BEST_PARAMS["RandomForest"]
    return RandomForestClassifier(**p, random_state=SEED)


def make_prev_xgb():
    p = PREV_BEST_PARAMS["XGBoost"]
    return XGBClassifier(**p, random_state=SEED, eval_metric="logloss")


def make_prev_lgbm():
    p = PREV_BEST_PARAMS["LightGBM"]
    return LGBMClassifier(**p, random_state=SEED, verbose=-1)


def make_prev_voting():
    return VotingClassifier(
        estimators=[
            ("logreg", make_prev_logreg()),
            ("rf", make_prev_rf()),
            ("xgb", make_prev_xgb()),
            ("lgbm", make_prev_lgbm()),
        ],
        voting="soft",
    )


prev_results = {}
for name, fn in [
    ("LogReg", make_prev_logreg),
    ("RF", make_prev_rf),
    ("XGB", make_prev_xgb),
    ("LGBM", make_prev_lgbm),
    ("Voting", make_prev_voting),
]:
    _, mean_m = cross_validate(fn, X_raw, y, feature_builder=fb)
    prev_results[name] = mean_m

print(f"\n{'Model':<20} {'Prev AUC':>12} {'New AUC':>12} {'Diff':>10}")
print("-" * 60)
for name in ["LogReg", "RF", "XGB", "LGBM", "Voting"]:
    prev_auc = prev_results[name]["auc"]
    new_auc = new_results[name]["auc"]
    diff = new_auc - prev_auc
    marker = " ***" if diff > 0.001 else ""
    print(f"  {name:<20} {prev_auc:>10.4f} {new_auc:>10.4f} {diff:>+8.4f}{marker}")


# ============================================================
# Phase 4: Generate Submissions (if improved)
# ============================================================
print("\n" + "=" * 60)
print("Phase 4: Submission Generation")
print("=" * 60)

sample_submit = pd.read_csv(SAMPLE_SUBMIT_CSV, index_col=0, header=None)

# Build features for full training and test data
X_tr_full, X_te_full = fb(X_raw, test)
print(f"  Full train features: {X_tr_full.shape}")
print(f"  Full test features:  {X_te_full.shape}")

# Determine which params are better
new_voting_auc = new_results["Voting"]["auc"]
prev_voting_auc = prev_results["Voting"]["auc"]
improved = new_voting_auc > prev_voting_auc

# Always generate submission with re-tuned params
model = make_new_voting()
model.fit(X_tr_full, y)
pred = model.predict_proba(X_te_full)[:, 1]
sub = sample_submit.copy()
sub[1] = pred
sub.to_csv("submit_retuned_voting.csv", header=None)
print(f"  submit_retuned_voting.csv: mean={pred.mean():.3f}, std={pred.std():.3f}")

# Also generate with previous params for comparison
model_prev = make_prev_voting()
model_prev.fit(X_tr_full, y)
pred_prev = model_prev.predict_proba(X_te_full)[:, 1]
sub_prev = sample_submit.copy()
sub_prev[1] = pred_prev
sub_prev.to_csv("submit_prev_params_voting.csv", header=None)
print(f"  submit_prev_params_voting.csv: mean={pred_prev.mean():.3f}, std={pred_prev.std():.3f}")

# Generate with best single model too
best_single_name = max(
    ["LogReg", "RF", "XGB", "LGBM"],
    key=lambda k: new_results[k]["auc"],
)
best_single_auc = new_results[best_single_name]["auc"]
print(f"\n  Best single model: {best_single_name} (AUC={best_single_auc:.4f})")

best_model_map = {
    "LogReg": make_new_logreg,
    "RF": make_new_rf,
    "XGB": make_new_xgb,
    "LGBM": make_new_lgbm,
}
best_single = best_model_map[best_single_name]()
best_single.fit(X_tr_full, y)
pred_bs = best_single.predict_proba(X_te_full)[:, 1]
sub_bs = sample_submit.copy()
sub_bs[1] = pred_bs
sub_bs.to_csv(f"submit_retuned_{best_single_name.lower()}.csv", header=None)
print(f"  submit_retuned_{best_single_name.lower()}.csv: mean={pred_bs.mean():.3f}, std={pred_bs.std():.3f}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

if improved:
    print(f"  Re-tuning IMPROVED Voting AUC: {prev_voting_auc:.4f} -> {new_voting_auc:.4f} (+{new_voting_auc - prev_voting_auc:.4f})")
    print(f"  Recommended submission: submit_retuned_voting.csv")
else:
    print(f"  Re-tuning did NOT improve Voting AUC: {prev_voting_auc:.4f} -> {new_voting_auc:.4f} ({new_voting_auc - prev_voting_auc:+.4f})")
    print(f"  Recommended submission: submit_prev_params_voting.csv")

print("\n  New BEST_PARAMS for domain+missing features:")
for name, params in best_params.items():
    print(f"    {name}: {params}")

print("\nDone!")
