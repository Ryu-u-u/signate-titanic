"""Experiment 31: Stability Feature Selection (Boruta-like).

Hypothesis: Features that are consistently important across multiple
seeds/folds are more robust and lead to better generalization.

Phase 1: Compute permutation importance across 5 seeds x 5 folds = 25 evaluations
Phase 2: Select stable features (important in >= 60% of evaluations)
Phase 3: Evaluate with stable features only vs all features
Phase 4: Generate submission if improved
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
from sklearn.inspection import permutation_importance
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
# Feature Builder: domain+missing
# ============================================================
fb = make_exp_builder(
    missing_flags=True, age_bins=None, fare_bins=None,
    interactions=False, polynomial=False, group_stats=False,
    freq_encoding=False, rank_features=False, domain_features=True,
)

# ============================================================
# BEST_PARAMS
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


# ============================================================
# Model Factory Functions
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
        ], voting="soft",
    )


# ============================================================
# Phase 1: Permutation Importance across Multiple Seeds/Folds
# ============================================================
print("=" * 60)
print("Phase 1: Permutation Importance (5 seeds x 5 folds = 25 evals)")
print("=" * 60)

EVAL_SEEDS = [42, 123, 456, 789, 2024]
PI_THRESHOLD = 0.001
N_REPEATS = 10

importance_counts = {}   # feature -> count of times PI > threshold
importance_values = {}   # feature -> list of PI values
total_evals = len(EVAL_SEEDS) * N_FOLDS

for seed_idx, eval_seed in enumerate(EVAL_SEEDS):
    cv = get_cv_splitter(n_folds=N_FOLDS, seed=eval_seed)
    print(f"\n  Seed {eval_seed} ({seed_idx + 1}/{len(EVAL_SEEDS)}):")

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_raw, y)):
        X_tr_raw, X_va_raw = X_raw.iloc[tr_idx], X_raw.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        X_tr, X_va = fb(X_tr_raw, X_va_raw)

        model = make_tuned_rf()
        model.fit(X_tr, y_tr)

        perm = permutation_importance(
            model, X_va, y_va,
            scoring="roc_auc", n_repeats=N_REPEATS,
            random_state=eval_seed,
        )

        for i, feat in enumerate(X_tr.columns):
            importance_counts.setdefault(feat, 0)
            importance_values.setdefault(feat, [])
            if perm.importances_mean[i] > PI_THRESHOLD:
                importance_counts[feat] += 1
            importance_values[feat].append(perm.importances_mean[i])

        print(f"    Fold {fold}: trained and evaluated PI")

# Print full importance summary
print(f"\n  Feature Importance Summary (threshold={PI_THRESHOLD}):")
print(f"  {'Feature':<30} {'Count':>6}/{total_evals}  {'Mean PI':>10} {'Std PI':>10} {'Stable':>7}")
print("  " + "-" * 70)

stability_threshold = int(total_evals * 0.6)  # 60% = 15/25

for feat in sorted(importance_counts.keys(),
                   key=lambda f: importance_counts.get(f, 0), reverse=True):
    count = importance_counts[feat]
    mean_pi = np.mean(importance_values[feat])
    std_pi = np.std(importance_values[feat])
    is_stable = count >= stability_threshold
    marker = "YES" if is_stable else "no"
    print(f"  {feat:<30} {count:>6}/{total_evals}  {mean_pi:>10.5f} {std_pi:>10.5f} {marker:>7}")


# ============================================================
# Phase 2: Select Stable Features
# ============================================================
print("\n" + "=" * 60)
print("Phase 2: Stable Feature Selection")
print("=" * 60)

stable_features = sorted([
    f for f, count in importance_counts.items()
    if count >= stability_threshold
])

all_features = sorted(importance_counts.keys())
dropped_features = sorted(set(all_features) - set(stable_features))

print(f"  Total features:    {len(all_features)}")
print(f"  Stable features:   {len(stable_features)} (>= {stability_threshold}/{total_evals} = 60%)")
print(f"  Dropped features:  {len(dropped_features)}")
print(f"\n  Stable: {stable_features}")
print(f"  Dropped: {dropped_features}")


def make_stable_builder():
    """Feature builder that keeps only stable features."""
    base_fb = make_exp_builder(
        missing_flags=True, age_bins=None, fare_bins=None,
        interactions=False, polynomial=False, group_stats=False,
        freq_encoding=False, rank_features=False, domain_features=True,
    )

    def builder(X_train_raw, X_val_raw):
        X_tr, X_va = base_fb(X_train_raw, X_val_raw)
        # Keep only stable features (handle case where feature might not exist)
        available_stable = [f for f in stable_features if f in X_tr.columns]
        X_tr = X_tr[available_stable]
        X_va = X_va[available_stable]
        return X_tr, X_va

    return builder


stable_fb = make_stable_builder()

# ============================================================
# Phase 3: Evaluate with Stable Features Only
# ============================================================
print("\n" + "=" * 60)
print("Phase 3: Comparison - All Features vs Stable Features Only")
print("=" * 60)

MODEL_FNS = {
    "LogReg": make_tuned_logreg,
    "RF": make_tuned_rf,
    "XGB": make_tuned_xgb,
    "LGBM": make_tuned_lgbm,
    "Voting": make_tuned_voting,
}

all_feat_results = {}
stable_feat_results = {}

print(f"\n  {'Model':<12} {'All AUC':>10} {'Stable AUC':>12} {'Diff':>8}")
print("  " + "-" * 46)

for name, model_fn in MODEL_FNS.items():
    # All features
    _, mean_all = cross_validate(model_fn, X_raw, y, feature_builder=fb)
    all_feat_results[name] = mean_all

    # Stable features only
    _, mean_stable = cross_validate(model_fn, X_raw, y, feature_builder=stable_fb)
    stable_feat_results[name] = mean_stable

    diff = mean_stable["auc"] - mean_all["auc"]
    marker = " ***" if diff > 0.001 else ""
    print(f"  {name:<12} {mean_all['auc']:>10.4f} {mean_stable['auc']:>12.4f} {diff:>+8.4f}{marker}")

# Additional metrics comparison for Voting
print(f"\n  Voting Detailed Metrics:")
print(f"    {'Metric':<12} {'All':>10} {'Stable':>10}")
print("    " + "-" * 34)
for metric in ["accuracy", "f1", "auc", "logloss"]:
    v_all = all_feat_results["Voting"][metric]
    v_stable = stable_feat_results["Voting"][metric]
    print(f"    {metric:<12} {v_all:>10.4f} {v_stable:>10.4f}")


# ============================================================
# Phase 4: Generate Submission if Improved
# ============================================================
print("\n" + "=" * 60)
print("Phase 4: Submission Generation")
print("=" * 60)

sample_submit = pd.read_csv(SAMPLE_SUBMIT_CSV, index_col=0, header=None)

voting_all_auc = all_feat_results["Voting"]["auc"]
voting_stable_auc = stable_feat_results["Voting"]["auc"]
improved = voting_stable_auc > voting_all_auc

# Submission with stable features
X_tr_stable, X_te_stable = stable_fb(X_raw, test)
print(f"  Stable train features: {X_tr_stable.shape}")
print(f"  Stable test features:  {X_te_stable.shape}")

model_stable = make_tuned_voting()
model_stable.fit(X_tr_stable, y)
pred_stable = model_stable.predict_proba(X_te_stable)[:, 1]

sub_stable = sample_submit.copy()
sub_stable[1] = pred_stable
sub_stable.to_csv("submit_stable_features_voting.csv", header=None)
print(f"  submit_stable_features_voting.csv: mean={pred_stable.mean():.3f}, std={pred_stable.std():.3f}")

# Submission with all features (baseline)
X_tr_full, X_te_full = fb(X_raw, test)
print(f"  All train features: {X_tr_full.shape}")
print(f"  All test features:  {X_te_full.shape}")

model_all = make_tuned_voting()
model_all.fit(X_tr_full, y)
pred_all = model_all.predict_proba(X_te_full)[:, 1]

sub_all = sample_submit.copy()
sub_all[1] = pred_all
sub_all.to_csv("submit_all_features_voting.csv", header=None)
print(f"  submit_all_features_voting.csv: mean={pred_all.mean():.3f}, std={pred_all.std():.3f}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

print(f"\n  Feature selection: {len(stable_features)}/{len(all_features)} features retained")
if dropped_features:
    print(f"  Dropped: {dropped_features}")

if improved:
    print(f"\n  Stable features IMPROVED Voting AUC: {voting_all_auc:.4f} -> {voting_stable_auc:.4f} (+{voting_stable_auc - voting_all_auc:.4f})")
    print(f"  Recommended submission: submit_stable_features_voting.csv")
else:
    print(f"\n  Stable features did NOT improve Voting AUC: {voting_all_auc:.4f} -> {voting_stable_auc:.4f} ({voting_stable_auc - voting_all_auc:+.4f})")
    print(f"  Recommended submission: submit_all_features_voting.csv")

print(f"\n  Stable features list (for use in other experiments):")
print(f"  stable_features = {stable_features}")

print("\nDone!")
