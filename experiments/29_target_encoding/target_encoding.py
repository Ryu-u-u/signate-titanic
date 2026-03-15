"""Experiment 29: CV-safe Target Encoding.

Hypothesis: Target encoding of low-cardinality combinations (pclass*sex,
embarked*sex, pclass*embarked) can capture conditional survival rates more
efficiently than one-hot encoding.

Phase 1: Implement leave-one-out target encoding with m-estimate smoothing
Phase 2: Create feature builder that adds target-encoded features
Phase 3: Evaluate with different m values (5, 10, 20, 50)
Phase 4: Compare best m-value results vs baseline
Phase 5: Generate submission if improved
"""
import sys
sys.path.append("../..")

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

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

# Target encoding combination specs:
#   name -> (col_a, col_b) from the processed (post-pipeline) features.
#   After build_pipeline, embarked becomes embarked_S, embarked_C, embarked_Q.
#   We use pclass and sex directly, and embarked_S as a proxy for embarked
#   (binary: 1=Southampton, 0=other).
TE_SPECS = [
    ("pclass_sex", ["pclass", "sex"]),
    ("embarked_sex", ["embarked_S", "sex"]),
    ("pclass_embarked", ["pclass", "embarked_S"]),
]

M_VALUES = [5, 10, 20, 50]

# ============================================================
# Data Loading
# ============================================================
print("=" * 60)
print("Experiment 29: CV-safe Target Encoding")
print("=" * 60)

train = pd.read_csv(TRAIN_CSV, index_col=0)
test = pd.read_csv(TEST_CSV, index_col=0)

y = train[TARGET_COL]
X_raw = train.drop(columns=[TARGET_COL])

print(f"Train: {X_raw.shape}, Test: {test.shape}")

# Feature builder: domain + missing (baseline)
fb_baseline = make_exp_builder(
    missing_flags=True, age_bins=None, fare_bins=None,
    interactions=False, polynomial=False, group_stats=False,
    freq_encoding=False, rank_features=False, domain_features=True,
)

# ============================================================
# Model Factories
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
# Phase 1: Leave-one-out Target Encoding
# ============================================================
print("\n" + "=" * 60)
print("Phase 1: LOO Target Encoding Implementation")
print("=" * 60)


def loo_target_encode(train_col, train_y, test_col, m=10):
    """Leave-one-out target encoding with m-estimate smoothing.

    Parameters
    ----------
    train_col : pd.Series
        Categorical column values for training data.
    train_y : pd.Series
        Target values for training data.
    test_col : pd.Series
        Categorical column values for test/validation data.
    m : float
        Smoothing parameter (higher = more regularization toward global mean).

    Returns
    -------
    encoded_train : pd.Series
        LOO-encoded training values.
    encoded_test : pd.Series
        Encoded test/validation values (using full training stats).
    """
    global_mean = train_y.mean()

    # For training: LOO to prevent target leakage
    encoded_train = pd.Series(0.0, index=train_col.index)
    for val in train_col.unique():
        mask = train_col == val
        n = mask.sum()
        group_sum = train_y[mask].sum()
        for idx in train_col[mask].index:
            # LOO: exclude current sample
            loo_sum = group_sum - train_y[idx]
            loo_n = n - 1
            encoded_train[idx] = (loo_sum + m * global_mean) / (loo_n + m)

    # For test: use full training statistics
    encoded_test = pd.Series(global_mean, index=test_col.index)
    stats = train_y.groupby(train_col).agg(["sum", "count"])
    for val in test_col.unique():
        mask = test_col == val
        if val in stats.index:
            s, n = stats.loc[val, "sum"], stats.loc[val, "count"]
            encoded_test[mask] = (s + m * global_mean) / (n + m)

    return encoded_train, encoded_test


# Quick sanity check
print("  LOO target encoding function defined.")
print(f"  Target encoding specs: {[s[0] for s in TE_SPECS]}")

# ============================================================
# Phase 2: Feature Builder with Target Encoding
# ============================================================
print("\n" + "=" * 60)
print("Phase 2: Target-Encoding Feature Builder")
print("=" * 60)


def make_te_builder(m=10):
    """Create a feature builder that adds LOO target-encoded features.

    The builder wraps the base domain+missing builder and appends
    target-encoded combination columns. Target encoding is computed
    within each CV fold to prevent leakage.

    Parameters
    ----------
    m : float
        Smoothing parameter for m-estimate regularization.

    Returns
    -------
    builder : callable
        (X_train_raw, X_val_raw) -> (X_train, X_val)
    """
    base_fb = make_exp_builder(
        missing_flags=True, age_bins=None, fare_bins=None,
        interactions=False, polynomial=False, group_stats=False,
        freq_encoding=False, rank_features=False, domain_features=True,
    )

    def builder(X_train_raw, X_val_raw):
        # Apply base feature pipeline
        X_tr, X_va = base_fb(X_train_raw, X_val_raw)

        # Get training target for this fold (using index alignment)
        y_tr = y.loc[X_train_raw.index]

        # Add target-encoded combination features
        for name, cols in TE_SPECS:
            # Create combination key from processed features
            tr_key = X_tr[cols[0]].astype(str) + "_" + X_tr[cols[1]].astype(str)
            va_key = X_va[cols[0]].astype(str) + "_" + X_va[cols[1]].astype(str)

            te_tr, te_va = loo_target_encode(tr_key, y_tr, va_key, m=m)
            X_tr[f"te_{name}"] = te_tr.values
            X_va[f"te_{name}"] = te_va.values

        return X_tr, X_va

    return builder


print(f"  make_te_builder defined with TE specs: {[s[0] for s in TE_SPECS]}")

# ============================================================
# Phase 3: Evaluate Different m Values
# ============================================================
print("\n" + "=" * 60)
print("Phase 3: m-value Search")
print(f"  m values: {M_VALUES}")
print("=" * 60)

models = {
    "LogReg": make_tuned_logreg,
    "RandomForest": make_tuned_rf,
    "XGBoost": make_tuned_xgb,
    "LightGBM": make_tuned_lgbm,
    "Voting": make_tuned_voting,
}

# Store results: results[m][model_name] = mean_metrics
te_results = {}

for m_val in M_VALUES:
    print(f"\n  --- m = {m_val} ---")
    te_fb = make_te_builder(m=m_val)
    te_results[m_val] = {}

    for name, model_fn in models.items():
        fold_metrics, mean_metrics = cross_validate(
            model_fn, X_raw, y, feature_builder=te_fb,
        )
        te_results[m_val][name] = mean_metrics
        print(f"    {name:15s}: AUC={mean_metrics['auc']:.6f}  "
              f"Acc={mean_metrics['accuracy']:.6f}")

# Find best m for Voting
best_m = max(M_VALUES, key=lambda m: te_results[m]["Voting"]["auc"])
print(f"\n  Best m (by Voting AUC): {best_m}")
print(f"  Voting AUC at best m: {te_results[best_m]['Voting']['auc']:.6f}")

# ============================================================
# Phase 4: Compare vs Baseline
# ============================================================
print("\n" + "=" * 60)
print("Phase 4: Target Encoding vs Baseline Comparison")
print("=" * 60)

# Baseline evaluation (domain+missing, no target encoding)
baseline_results = {}
print("\n  Baseline (domain+missing, no TE):")
for name, model_fn in models.items():
    _, mean_metrics = cross_validate(
        model_fn, X_raw, y, feature_builder=fb_baseline,
    )
    baseline_results[name] = mean_metrics
    print(f"    {name:15s}: AUC={mean_metrics['auc']:.6f}  "
          f"Acc={mean_metrics['accuracy']:.6f}")

print(f"\n  Best TE (m={best_m}):")
for name in models:
    print(f"    {name:15s}: AUC={te_results[best_m][name]['auc']:.6f}  "
          f"Acc={te_results[best_m][name]['accuracy']:.6f}")

print(f"\n  Delta (TE - Baseline):")
print(f"  {'Model':15s}  {'Baseline AUC':>14s}  {'TE AUC':>14s}  {'Delta':>10s}")
print("-" * 60)
for name in models:
    bl_auc = baseline_results[name]["auc"]
    te_auc = te_results[best_m][name]["auc"]
    delta = te_auc - bl_auc
    marker = " *" if delta > 0 else ""
    print(f"  {name:15s}  {bl_auc:14.6f}  {te_auc:14.6f}  {delta:+10.6f}{marker}")

# m-value sensitivity table
print(f"\n  m-value sensitivity (Voting AUC):")
for m_val in M_VALUES:
    auc = te_results[m_val]["Voting"]["auc"]
    print(f"    m={m_val:3d}: AUC={auc:.6f}")

# ============================================================
# Phase 5: Generate Submission
# ============================================================
print("\n" + "=" * 60)
print("Phase 5: Submission Generation")
print("=" * 60)

bl_voting_auc = baseline_results["Voting"]["auc"]
te_voting_auc = te_results[best_m]["Voting"]["auc"]

if te_voting_auc > bl_voting_auc:
    print(f"  TE (m={best_m}) improves Voting AUC: "
          f"{bl_voting_auc:.6f} -> {te_voting_auc:.6f}")
    print("  Generating TE submission.")
else:
    print(f"  TE does not improve Voting AUC: "
          f"{bl_voting_auc:.6f} -> {te_voting_auc:.6f}")
    print("  Generating submission anyway for comparison.")

# For submission: train on full X_raw, predict on test.
# Target encoding uses full training y (no LOO needed for test).
train_stats = compute_train_stats(X_raw)

# Build features for full train and test using base pipeline
base_fb_full = make_exp_builder(
    missing_flags=True, age_bins=None, fare_bins=None,
    interactions=False, polynomial=False, group_stats=False,
    freq_encoding=False, rank_features=False, domain_features=True,
)
X_full, X_test_feat = base_fb_full(X_raw, test)

# Add target-encoded features for submission (full train -> test)
for name, cols in TE_SPECS:
    tr_key = X_full[cols[0]].astype(str) + "_" + X_full[cols[1]].astype(str)
    te_key = X_test_feat[cols[0]].astype(str) + "_" + X_test_feat[cols[1]].astype(str)

    # For full training: use m-estimate (not LOO, since we are not validating)
    global_mean = y.mean()
    stats = y.groupby(tr_key).agg(["sum", "count"])

    # Train encoding (full, no LOO since there is no validation concern)
    te_tr = pd.Series(global_mean, index=tr_key.index)
    for val in tr_key.unique():
        mask = tr_key == val
        if val in stats.index:
            s, n = stats.loc[val, "sum"], stats.loc[val, "count"]
            te_tr[mask] = (s + best_m * global_mean) / (n + best_m)

    # Test encoding
    te_te = pd.Series(global_mean, index=te_key.index)
    for val in te_key.unique():
        mask = te_key == val
        if val in stats.index:
            s, n = stats.loc[val, "sum"], stats.loc[val, "count"]
            te_te[mask] = (s + best_m * global_mean) / (n + best_m)

    X_full[f"te_{name}"] = te_tr.values
    X_test_feat[f"te_{name}"] = te_te.values

print(f"  Full train features: {X_full.shape}")
print(f"  Test features:       {X_test_feat.shape}")

# Train and predict
model = make_tuned_voting()
model.fit(X_full, y)
proba = model.predict_proba(X_test_feat)[:, 1]

# Save submission
sample = pd.read_csv(SAMPLE_SUBMIT_CSV, index_col=0, header=None)
submission = pd.DataFrame({0: sample.index, 1: proba})
submission.to_csv(
    "submission_te.csv",
    index=False, header=False,
)
print("  Saved: submission_te.csv")

# Also generate baseline submission for reference
X_full_bl, X_test_bl = base_fb_full(X_raw, test)
model_bl = make_tuned_voting()
model_bl.fit(X_full_bl, y)
proba_bl = model_bl.predict_proba(X_test_bl)[:, 1]

sub_bl = pd.DataFrame({0: sample.index, 1: proba_bl})
sub_bl.to_csv(
    "submission_baseline.csv",
    index=False, header=False,
)
print("  Saved: submission_baseline.csv")

print("\n" + "=" * 60)
print("Experiment 29 complete!")
print("=" * 60)
