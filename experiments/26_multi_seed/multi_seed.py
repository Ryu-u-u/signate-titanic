"""Multi-Seed Averaging for Variance Reduction.

Hypothesis: Using multiple seeds for model random_state and averaging
predictions reduces variance without biasing the estimate.

Phase 1: Evaluate each seed individually to observe variance
Phase 2: Custom multi-seed CV loop (average predictions across seeds per fold)
Phase 3: Generate submission with all seeds trained on full data
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
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import TRAIN_CSV, TEST_CSV, SAMPLE_SUBMIT_CSV, SEED, N_FOLDS, TARGET_COL
from src.utils import seed_everything
from src.exp_features import make_exp_builder
from src.evaluation import cross_validate, get_cv_splitter

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
# Configuration
# ============================================================
SEEDS = [42, 123, 456, 789, 2024]

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

# Feature builder: domain+missing
fb = make_exp_builder(
    missing_flags=True, age_bins=None, fare_bins=None,
    interactions=False, polynomial=False, group_stats=False,
    freq_encoding=False, rank_features=False, domain_features=True,
)


# ============================================================
# Model factories with configurable seed
# ============================================================
def make_logreg_with_seed(seed):
    p = BEST_PARAMS["LogReg"]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=p["C"], solver=p["solver"],
            max_iter=2000, random_state=seed,
        )),
    ])


def make_rf_with_seed(seed):
    p = BEST_PARAMS["RandomForest"]
    return RandomForestClassifier(**p, random_state=seed)


def make_xgb_with_seed(seed):
    p = BEST_PARAMS["XGBoost"]
    return XGBClassifier(**p, random_state=seed, eval_metric="logloss")


def make_lgbm_with_seed(seed):
    p = BEST_PARAMS["LightGBM"]
    return LGBMClassifier(**p, random_state=seed, verbose=-1)


def make_voting_with_seed(seed):
    return VotingClassifier(
        estimators=[
            ("logreg", make_logreg_with_seed(seed)),
            ("rf", make_rf_with_seed(seed)),
            ("xgb", make_xgb_with_seed(seed)),
            ("lgbm", make_lgbm_with_seed(seed)),
        ],
        voting="soft",
    )


# Fixed-seed factories for cross_validate compatibility
def make_tuned_logreg():
    return make_logreg_with_seed(SEED)


def make_tuned_rf():
    return make_rf_with_seed(SEED)


def make_tuned_xgb():
    return make_xgb_with_seed(SEED)


def make_tuned_lgbm():
    return make_lgbm_with_seed(SEED)


def make_tuned_voting():
    return make_voting_with_seed(SEED)


# ============================================================
# Phase 1: Evaluate each seed individually
# ============================================================
print("=" * 60)
print("Phase 1: Individual Seed Evaluation")
print("=" * 60)

# CV splits always use seed=42 for fair comparison
seed_results = {}

for s in SEEDS:
    def _make_voting(seed=s):
        return make_voting_with_seed(seed)

    _, mean_m = cross_validate(
        _make_voting, X_raw, y, n_folds=N_FOLDS, seed=SEED, feature_builder=fb
    )
    seed_results[s] = mean_m
    print(f"  Seed {s:>4d}: AUC={mean_m['auc']:.4f}, Acc={mean_m['accuracy']:.4f}")

# Variance analysis
aucs = [seed_results[s]["auc"] for s in SEEDS]
print(f"\n  Mean AUC:  {np.mean(aucs):.4f}")
print(f"  Std AUC:   {np.std(aucs):.4f}")
print(f"  Min AUC:   {np.min(aucs):.4f}")
print(f"  Max AUC:   {np.max(aucs):.4f}")
print(f"  Range:     {np.max(aucs) - np.min(aucs):.4f}")


# ============================================================
# Phase 2: Multi-Seed Average CV Evaluation
# ============================================================
print("\n" + "=" * 60)
print("Phase 2: Multi-Seed Averaged Predictions (Custom CV)")
print("=" * 60)

cv = get_cv_splitter(n_folds=N_FOLDS, seed=SEED)
fold_aucs_multi = []
fold_aucs_single = []

for fold, (tr_idx, va_idx) in enumerate(cv.split(X_raw, y)):
    X_tr_raw, X_va_raw = X_raw.iloc[tr_idx], X_raw.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # Build features once per fold
    X_tr, X_va = fb(X_tr_raw, X_va_raw)

    # Multi-seed averaging
    probas = []
    for s in SEEDS:
        model = make_voting_with_seed(s)
        model.fit(X_tr, y_tr)
        probas.append(model.predict_proba(X_va)[:, 1])
    avg_proba = np.mean(probas, axis=0)
    fold_auc_multi = roc_auc_score(y_va, avg_proba)
    fold_aucs_multi.append(fold_auc_multi)

    # Single-seed reference (seed=42)
    model_single = make_voting_with_seed(SEED)
    model_single.fit(X_tr, y_tr)
    proba_single = model_single.predict_proba(X_va)[:, 1]
    fold_auc_single = roc_auc_score(y_va, proba_single)
    fold_aucs_single.append(fold_auc_single)

    print(f"  Fold {fold}: Single={fold_auc_single:.4f}, Multi-Seed={fold_auc_multi:.4f}, "
          f"Diff={fold_auc_multi - fold_auc_single:+.4f}")

mean_multi = np.mean(fold_aucs_multi)
mean_single = np.mean(fold_aucs_single)
std_multi = np.std(fold_aucs_multi)
std_single = np.std(fold_aucs_single)

print(f"\n  Single-seed (42):  AUC={mean_single:.4f} (+/- {std_single:.4f})")
print(f"  Multi-seed avg:    AUC={mean_multi:.4f} (+/- {std_multi:.4f})")
print(f"  Difference:        {mean_multi - mean_single:+.4f}")
print(f"  Variance reduction: {std_single:.4f} -> {std_multi:.4f}")


# ============================================================
# Phase 2.5: Per-Model Multi-Seed Analysis
# ============================================================
print("\n" + "=" * 60)
print("Phase 2.5: Per-Model Multi-Seed Variance")
print("=" * 60)

model_factories = {
    "LogReg": make_logreg_with_seed,
    "RF": make_rf_with_seed,
    "XGB": make_xgb_with_seed,
    "LGBM": make_lgbm_with_seed,
    "Voting": make_voting_with_seed,
}

for model_name, factory in model_factories.items():
    model_seed_aucs = []
    for s in SEEDS:
        def _make_model(seed=s):
            return factory(seed)

        _, mean_m = cross_validate(
            _make_model, X_raw, y, n_folds=N_FOLDS, seed=SEED, feature_builder=fb
        )
        model_seed_aucs.append(mean_m["auc"])

    print(f"  {model_name:<10}: mean={np.mean(model_seed_aucs):.4f}, "
          f"std={np.std(model_seed_aucs):.4f}, "
          f"range={np.max(model_seed_aucs) - np.min(model_seed_aucs):.4f}")


# ============================================================
# Phase 3: Generate Submission with Multi-Seed Averaging
# ============================================================
print("\n" + "=" * 60)
print("Phase 3: Submission Generation")
print("=" * 60)

sample_submit = pd.read_csv(SAMPLE_SUBMIT_CSV, index_col=0, header=None)

# Build features for full train and test
X_tr_full, X_te_full = fb(X_raw, test)
print(f"  Full train features: {X_tr_full.shape}")
print(f"  Full test features:  {X_te_full.shape}")

# Multi-seed averaged predictions
all_test_probas = []
for s in SEEDS:
    model = make_voting_with_seed(s)
    model.fit(X_tr_full, y)
    pred = model.predict_proba(X_te_full)[:, 1]
    all_test_probas.append(pred)
    print(f"  Seed {s:>4d}: mean={pred.mean():.3f}, std={pred.std():.3f}")

avg_test_pred = np.mean(all_test_probas, axis=0)
print(f"\n  Multi-seed avg: mean={avg_test_pred.mean():.3f}, std={avg_test_pred.std():.3f}")

# Submission: multi-seed averaged
sub_multi = sample_submit.copy()
sub_multi[1] = avg_test_pred
sub_multi.to_csv("submit_multi_seed_voting.csv", header=None)
print(f"  submit_multi_seed_voting.csv saved.")

# Submission: single-seed reference
model_single = make_voting_with_seed(SEED)
model_single.fit(X_tr_full, y)
pred_single = model_single.predict_proba(X_te_full)[:, 1]
sub_single = sample_submit.copy()
sub_single[1] = pred_single
sub_single.to_csv("submit_single_seed_voting.csv", header=None)
print(f"  submit_single_seed_voting.csv saved.")

# Prediction correlation between single and multi-seed
corr = np.corrcoef(pred_single, avg_test_pred)[0, 1]
max_diff = np.max(np.abs(pred_single - avg_test_pred))
print(f"\n  Correlation (single vs multi): {corr:.6f}")
print(f"  Max absolute difference:       {max_diff:.6f}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

print(f"  Seeds used: {SEEDS}")
print(f"  CV AUC (single, seed=42): {mean_single:.4f} (+/- {std_single:.4f})")
print(f"  CV AUC (multi-seed avg):  {mean_multi:.4f} (+/- {std_multi:.4f})")
print(f"  AUC improvement:          {mean_multi - mean_single:+.4f}")
print(f"  Fold std reduction:       {std_single:.4f} -> {std_multi:.4f}")

if mean_multi > mean_single:
    print(f"\n  Multi-seed averaging IMPROVED AUC.")
    print(f"  Recommended submission: submit_multi_seed_voting.csv")
else:
    print(f"\n  Multi-seed averaging did NOT improve mean AUC.")
    print(f"  However, lower variance may still improve generalization.")
    print(f"  Consider submitting both files to compare Public scores.")

print("\nDone!")
