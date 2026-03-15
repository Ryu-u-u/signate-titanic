"""Experiment 32: Agreement-Gated Pseudo Labeling.

Hypothesis: Test samples where all models agree with high confidence can serve
as pseudo-labels, effectively expanding training data.

Phase 1: Train models on full training data, predict test
Phase 2: Identify high-agreement samples across all models
Phase 3: Augment training data with pseudo-labels and re-evaluate
Phase 4: Try different agreement thresholds (0.90, 0.95, 0.99)
Phase 5: Compare with baseline and generate submission if improved
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
from src.evaluation import cross_validate, get_cv_splitter
from src.exp_features import make_exp_builder

seed_everything(SEED)

# ============================================================
# Data Loading
# ============================================================
train = pd.read_csv(TRAIN_CSV, index_col=0)
test = pd.read_csv(TEST_CSV, index_col=0)

y = train[TARGET_COL]
X_raw = train.drop(columns=[TARGET_COL])

print(f"Train: {X_raw.shape}, Test: {test.shape}")

# ============================================================
# Configuration
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

THRESHOLDS = [0.90, 0.95, 0.99]

# Feature builder: domain+missing
fb = make_exp_builder(
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


model_fns = {
    "LogReg": make_tuned_logreg,
    "RandomForest": make_tuned_rf,
    "XGBoost": make_tuned_xgb,
    "LightGBM": make_tuned_lgbm,
}


# ============================================================
# Phase 0: Baseline CV (Voting, no pseudo-labels)
# ============================================================
print("=" * 60)
print("Experiment 32: Agreement-Gated Pseudo Labeling")
print("=" * 60)

print("\n--- Phase 0: Baseline CV (Voting, no pseudo-labels) ---")
_, baseline_mean = cross_validate(
    make_tuned_voting, X_raw, y, n_folds=N_FOLDS, seed=SEED, feature_builder=fb,
)
baseline_auc = baseline_mean["auc"]
print(f"  Baseline Voting AUC: {baseline_auc:.4f}")


# ============================================================
# Phase 1: Train models on full training data, predict test
# ============================================================
print("\n--- Phase 1: Train all models on full data, predict test ---")

X_tr_full, X_te_full = fb(X_raw, test)
print(f"  Train features: {X_tr_full.shape}")
print(f"  Test features:  {X_te_full.shape}")

test_probas = {}
for name, fn in model_fns.items():
    model = fn()
    model.fit(X_tr_full, y)
    proba = model.predict_proba(X_te_full)[:, 1]
    test_probas[name] = proba
    print(f"  {name:15s}: mean={proba.mean():.4f}, std={proba.std():.4f}")


# ============================================================
# Phase 2: Identify high-agreement samples
# ============================================================
print("\n--- Phase 2: High-Agreement Analysis ---")

all_probas = np.column_stack(list(test_probas.values()))
print(f"  Probas matrix shape: {all_probas.shape}")
print(f"  Model agreement correlation:")
for i, name_i in enumerate(test_probas.keys()):
    for j, name_j in enumerate(test_probas.keys()):
        if j > i:
            corr = np.corrcoef(all_probas[:, i], all_probas[:, j])[0, 1]
            print(f"    {name_i} vs {name_j}: {corr:.4f}")

for threshold in THRESHOLDS:
    high_pos = np.all(all_probas > threshold, axis=1)
    high_neg = np.all(all_probas < (1.0 - threshold), axis=1)
    pseudo_mask = high_pos | high_neg
    print(f"\n  Threshold {threshold}:")
    print(f"    High confidence positives: {high_pos.sum()}")
    print(f"    High confidence negatives: {high_neg.sum()}")
    print(f"    Total pseudo-labeled: {pseudo_mask.sum()} / {len(test)}")


# ============================================================
# Phase 3 & 4: Augment training data and re-evaluate
#   at different thresholds
# ============================================================
print("\n--- Phase 3 & 4: Pseudo-Label Augmented CV ---")

cv = get_cv_splitter(n_folds=N_FOLDS, seed=SEED)

results_by_threshold = {}

for threshold in THRESHOLDS:
    print(f"\n  === Threshold: {threshold} ===")

    # Compute pseudo-labels and mask based on full-data model agreement
    high_pos = np.all(all_probas > threshold, axis=1)
    high_neg = np.all(all_probas < (1.0 - threshold), axis=1)
    pseudo_mask = high_pos | high_neg
    pseudo_labels = (np.mean(all_probas, axis=1) > 0.5).astype(int)

    n_pseudo = pseudo_mask.sum()
    if n_pseudo == 0:
        print(f"    No pseudo-labeled samples at threshold {threshold}. Skipping.")
        results_by_threshold[threshold] = {
            "mean_auc": baseline_auc,
            "std_auc": 0.0,
            "n_pseudo": 0,
        }
        continue

    print(f"    Pseudo-labeled samples: {n_pseudo}")
    print(f"    Pseudo positive ratio: "
          f"{pseudo_labels[pseudo_mask].mean():.3f}")

    # Get the test rows that are pseudo-labeled
    test_pseudo = test[pseudo_mask]
    y_pseudo = pd.Series(
        pseudo_labels[pseudo_mask], name=y.name, index=test_pseudo.index,
    )

    fold_aucs = []
    cv = get_cv_splitter(n_folds=N_FOLDS, seed=SEED)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_raw, y)):
        X_tr_raw = X_raw.iloc[tr_idx]
        X_va_raw = X_raw.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        # Build features for train and validation folds
        X_tr, X_va = fb(X_tr_raw, X_va_raw)

        # Build features for pseudo-labeled test samples using train-fold stats
        _, X_pseudo = fb(X_tr_raw, test_pseudo)

        # Augment training data with pseudo-labeled samples
        X_tr_aug = pd.concat([X_tr, X_pseudo], ignore_index=True)
        y_tr_aug = pd.concat(
            [y_tr, y_pseudo.reset_index(drop=True)], ignore_index=True,
        )

        model = make_tuned_voting()
        model.fit(X_tr_aug, y_tr_aug)
        proba = model.predict_proba(X_va)[:, 1]
        fold_auc = roc_auc_score(y_va, proba)
        fold_aucs.append(fold_auc)

        print(f"    Fold {fold}: AUC={fold_auc:.4f} "
              f"(train: {len(X_tr)}+{len(X_pseudo)}={len(X_tr_aug)})")

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    diff = mean_auc - baseline_auc

    results_by_threshold[threshold] = {
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "n_pseudo": n_pseudo,
    }

    print(f"\n    Mean AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")
    print(f"    vs Baseline: {diff:+.4f}")


# ============================================================
# Phase 5: Comparison and Submission
# ============================================================
print("\n" + "=" * 60)
print("Phase 5: Comparison and Submission")
print("=" * 60)

print(f"\n  Baseline Voting AUC: {baseline_auc:.4f}")
print(f"\n  {'Threshold':>10s}  {'N_pseudo':>8s}  {'AUC':>8s}  {'Diff':>8s}")
print("  " + "-" * 42)

best_threshold = None
best_auc = baseline_auc

for threshold in THRESHOLDS:
    r = results_by_threshold[threshold]
    diff = r["mean_auc"] - baseline_auc
    marker = " *" if r["mean_auc"] > best_auc else ""
    print(f"  {threshold:10.2f}  {r['n_pseudo']:8d}  "
          f"{r['mean_auc']:8.4f}  {diff:+8.4f}{marker}")
    if r["mean_auc"] > best_auc:
        best_auc = r["mean_auc"]
        best_threshold = threshold

# Generate submission
sample_submit = pd.read_csv(SAMPLE_SUBMIT_CSV, index_col=0, header=None)

if best_threshold is not None:
    print(f"\n  Best threshold: {best_threshold} (AUC={best_auc:.4f})")
    print("  Generating pseudo-label augmented submission...")

    # Recompute pseudo-labels for best threshold
    high_pos = np.all(all_probas > best_threshold, axis=1)
    high_neg = np.all(all_probas < (1.0 - best_threshold), axis=1)
    pseudo_mask_best = high_pos | high_neg
    pseudo_labels_best = (np.mean(all_probas, axis=1) > 0.5).astype(int)

    test_pseudo_best = test[pseudo_mask_best]
    y_pseudo_best = pd.Series(
        pseudo_labels_best[pseudo_mask_best], name=y.name,
        index=test_pseudo_best.index,
    )

    # Train on full augmented data
    X_tr_full, X_te_full = fb(X_raw, test)
    _, X_pseudo_full = fb(X_raw, test_pseudo_best)

    X_aug_full = pd.concat([X_tr_full, X_pseudo_full], ignore_index=True)
    y_aug_full = pd.concat(
        [y, y_pseudo_best.reset_index(drop=True)], ignore_index=True,
    )

    print(f"  Augmented training size: {len(X_aug_full)} "
          f"({len(X_tr_full)} + {len(X_pseudo_full)})")

    model = make_tuned_voting()
    model.fit(X_aug_full, y_aug_full)
    test_pred = model.predict_proba(X_te_full)[:, 1]

    submission = sample_submit.copy()
    submission[1] = test_pred
    submission.to_csv("submit_pseudo_labeling.csv", header=None)
    print("  Saved: submit_pseudo_labeling.csv")
else:
    print("\n  No threshold improved over baseline.")
    print("  Generating baseline submission (no pseudo-labels)...")

    X_tr_full, X_te_full = fb(X_raw, test)
    model = make_tuned_voting()
    model.fit(X_tr_full, y)
    test_pred = model.predict_proba(X_te_full)[:, 1]

    submission = sample_submit.copy()
    submission[1] = test_pred
    submission.to_csv("submit_pseudo_labeling_baseline.csv", header=None)
    print("  Saved: submit_pseudo_labeling_baseline.csv")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

print(f"  Baseline Voting CV AUC:  {baseline_auc:.4f}")
if best_threshold is not None:
    print(f"  Best pseudo-label AUC:   {best_auc:.4f} (threshold={best_threshold})")
    print(f"  Improvement:             {best_auc - baseline_auc:+.4f}")
    print(f"  Pseudo-labeled samples:  {results_by_threshold[best_threshold]['n_pseudo']}")
else:
    print("  Pseudo-labeling did NOT improve over baseline at any threshold.")
    print("  This suggests the models already capture the easy patterns well.")

print("\nDone!")
