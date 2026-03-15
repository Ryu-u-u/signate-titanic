"""Experiment 33: Tabular Augmentation (Mixup).

Hypothesis: Conservative data augmentation via Mixup in the minority/uncertain
regions can help with only 445 training samples.

Phase 1: Implement Mixup-based tabular augmentation
Phase 2: Evaluate augmented training within CV at different n_aug levels
Phase 3: Per-model analysis of augmentation sensitivity
Phase 4: Compare with baseline
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
print(f"Class distribution: {y.value_counts().to_dict()}")

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

N_AUG_VALUES = [50, 100, 200]
ALPHA_VALUES = [0.1, 0.2, 0.4]

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


# ============================================================
# Mixup Augmentation
# ============================================================
def mixup_augment(X, y, n_aug=100, alpha=0.2, seed=42):
    """Generate synthetic samples via Mixup within same class.

    For each class, randomly pick pairs and interpolate:
        x_new = lam * x_i + (1 - lam) * x_j
    where lam ~ Beta(alpha, alpha).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels (0 or 1).
    n_aug : int
        Number of augmented samples per class.
    alpha : float
        Beta distribution parameter (smaller = closer to originals).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X_aug : pd.DataFrame
        Augmented feature matrix.
    y_aug : pd.Series
        Augmented labels.
    """
    rng = np.random.RandomState(seed)
    X_aug_list = []
    y_aug_list = []

    for label in [0, 1]:
        mask = y == label
        X_class = X[mask].values
        n_class = len(X_class)

        if n_class < 2:
            continue

        for _ in range(n_aug):
            i, j = rng.randint(0, n_class, 2)
            # Ensure different samples
            while j == i and n_class > 1:
                j = rng.randint(0, n_class)
            lam = rng.beta(alpha, alpha)
            x_new = lam * X_class[i] + (1 - lam) * X_class[j]
            X_aug_list.append(x_new)
            y_aug_list.append(label)

    X_aug = pd.DataFrame(X_aug_list, columns=X.columns)
    y_aug = pd.Series(y_aug_list, name=y.name)
    return X_aug, y_aug


# ============================================================
# Phase 0: Baseline CV
# ============================================================
print("=" * 60)
print("Experiment 33: Tabular Augmentation (Mixup)")
print("=" * 60)

print("\n--- Phase 0: Baseline CV (no augmentation) ---")
_, baseline_mean = cross_validate(
    make_tuned_voting, X_raw, y, n_folds=N_FOLDS, seed=SEED, feature_builder=fb,
)
baseline_auc = baseline_mean["auc"]
print(f"  Baseline Voting AUC: {baseline_auc:.4f}")


# ============================================================
# Phase 1: Augmentation Analysis
# ============================================================
print("\n--- Phase 1: Augmentation Sample Statistics ---")

# Build features once for analysis
X_tr_demo, _ = fb(X_raw, test)

X_aug_demo, y_aug_demo = mixup_augment(
    X_tr_demo, y, n_aug=100, alpha=0.2, seed=SEED,
)
print(f"  Original train: {X_tr_demo.shape}")
print(f"  Augmented samples: {X_aug_demo.shape}")
print(f"  Augmented class distribution: "
      f"{pd.Series(y_aug_demo).value_counts().to_dict()}")

# Compare feature distributions
print("\n  Feature statistics (original vs augmented):")
print(f"  {'Feature':>20s}  {'orig_mean':>10s}  {'aug_mean':>10s}  "
      f"{'orig_std':>10s}  {'aug_std':>10s}")
print("  " + "-" * 65)
for col in X_tr_demo.columns[:10]:  # Show first 10 features
    orig_mean = X_tr_demo[col].mean()
    aug_mean = X_aug_demo[col].mean()
    orig_std = X_tr_demo[col].std()
    aug_std = X_aug_demo[col].std()
    print(f"  {col:>20s}  {orig_mean:10.4f}  {aug_mean:10.4f}  "
          f"{orig_std:10.4f}  {aug_std:10.4f}")


# ============================================================
# Phase 2: Evaluate augmented training within CV
# ============================================================
print("\n--- Phase 2: Mixup Augmented CV (varying n_aug, alpha=0.2) ---")

results_n_aug = {}

for n_aug in N_AUG_VALUES:
    fold_aucs = []
    cv = get_cv_splitter(n_folds=N_FOLDS, seed=SEED)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_raw, y)):
        X_tr_raw = X_raw.iloc[tr_idx]
        X_va_raw = X_raw.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        # Build features
        X_tr, X_va = fb(X_tr_raw, X_va_raw)

        # Augment training data with Mixup
        X_aug, y_aug = mixup_augment(
            X_tr, y_tr, n_aug=n_aug, alpha=0.2, seed=SEED + fold,
        )
        X_tr_full = pd.concat([X_tr, X_aug], ignore_index=True)
        y_tr_full = pd.concat([y_tr, y_aug], ignore_index=True)

        model = make_tuned_voting()
        model.fit(X_tr_full, y_tr_full)
        proba = model.predict_proba(X_va)[:, 1]
        fold_auc = roc_auc_score(y_va, proba)
        fold_aucs.append(fold_auc)

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    diff = mean_auc - baseline_auc

    results_n_aug[n_aug] = {
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "fold_aucs": fold_aucs,
    }
    print(f"  n_aug={n_aug:>3d}: AUC={mean_auc:.4f} (+/- {std_auc:.4f})  "
          f"diff={diff:+.4f}")


# ============================================================
# Phase 2b: Vary alpha (Mixup interpolation strength)
# ============================================================
print("\n--- Phase 2b: Vary Alpha (n_aug=100) ---")

# Find best n_aug from Phase 2
best_n_aug = max(results_n_aug, key=lambda k: results_n_aug[k]["mean_auc"])
print(f"  Best n_aug from Phase 2: {best_n_aug}")

results_alpha = {}

for alpha in ALPHA_VALUES:
    fold_aucs = []
    cv = get_cv_splitter(n_folds=N_FOLDS, seed=SEED)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_raw, y)):
        X_tr_raw = X_raw.iloc[tr_idx]
        X_va_raw = X_raw.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        X_tr, X_va = fb(X_tr_raw, X_va_raw)

        X_aug, y_aug = mixup_augment(
            X_tr, y_tr, n_aug=100, alpha=alpha, seed=SEED + fold,
        )
        X_tr_full = pd.concat([X_tr, X_aug], ignore_index=True)
        y_tr_full = pd.concat([y_tr, y_aug], ignore_index=True)

        model = make_tuned_voting()
        model.fit(X_tr_full, y_tr_full)
        proba = model.predict_proba(X_va)[:, 1]
        fold_auc = roc_auc_score(y_va, proba)
        fold_aucs.append(fold_auc)

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    diff = mean_auc - baseline_auc

    results_alpha[alpha] = {
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "fold_aucs": fold_aucs,
    }
    print(f"  alpha={alpha:.1f}: AUC={mean_auc:.4f} (+/- {std_auc:.4f})  "
          f"diff={diff:+.4f}")


# ============================================================
# Phase 3: Per-Model Augmentation Sensitivity
# ============================================================
print("\n--- Phase 3: Per-Model Augmentation Sensitivity ---")

individual_models = {
    "LogReg": make_tuned_logreg,
    "RandomForest": make_tuned_rf,
    "XGBoost": make_tuned_xgb,
    "LightGBM": make_tuned_lgbm,
    "Voting": make_tuned_voting,
}

# Use best alpha from Phase 2b
best_alpha = max(results_alpha, key=lambda k: results_alpha[k]["mean_auc"])
print(f"  Using n_aug={best_n_aug}, alpha={best_alpha}")

for model_name, model_fn in individual_models.items():
    fold_aucs_base = []
    fold_aucs_aug = []
    cv = get_cv_splitter(n_folds=N_FOLDS, seed=SEED)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_raw, y)):
        X_tr_raw = X_raw.iloc[tr_idx]
        X_va_raw = X_raw.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        X_tr, X_va = fb(X_tr_raw, X_va_raw)

        # Baseline (no augmentation)
        model_base = model_fn()
        model_base.fit(X_tr, y_tr)
        proba_base = model_base.predict_proba(X_va)[:, 1]
        fold_aucs_base.append(roc_auc_score(y_va, proba_base))

        # Augmented
        X_aug, y_aug = mixup_augment(
            X_tr, y_tr, n_aug=best_n_aug, alpha=best_alpha, seed=SEED + fold,
        )
        X_tr_full = pd.concat([X_tr, X_aug], ignore_index=True)
        y_tr_full = pd.concat([y_tr, y_aug], ignore_index=True)

        model_aug = model_fn()
        model_aug.fit(X_tr_full, y_tr_full)
        proba_aug = model_aug.predict_proba(X_va)[:, 1]
        fold_aucs_aug.append(roc_auc_score(y_va, proba_aug))

    base_mean = np.mean(fold_aucs_base)
    aug_mean = np.mean(fold_aucs_aug)
    diff = aug_mean - base_mean
    marker = " *" if diff > 0 else ""
    print(f"  {model_name:15s}: base={base_mean:.4f}, aug={aug_mean:.4f}, "
          f"diff={diff:+.4f}{marker}")


# ============================================================
# Phase 4: Comparison Summary
# ============================================================
print("\n" + "=" * 60)
print("Phase 4: Comparison Summary")
print("=" * 60)

print(f"\n  Baseline Voting AUC: {baseline_auc:.4f}")

print(f"\n  By n_aug (alpha=0.2):")
print(f"  {'n_aug':>6s}  {'AUC':>8s}  {'Diff':>8s}")
print("  " + "-" * 28)
for n_aug in N_AUG_VALUES:
    r = results_n_aug[n_aug]
    diff = r["mean_auc"] - baseline_auc
    print(f"  {n_aug:6d}  {r['mean_auc']:8.4f}  {diff:+8.4f}")

print(f"\n  By alpha (n_aug=100):")
print(f"  {'alpha':>6s}  {'AUC':>8s}  {'Diff':>8s}")
print("  " + "-" * 28)
for alpha in ALPHA_VALUES:
    r = results_alpha[alpha]
    diff = r["mean_auc"] - baseline_auc
    print(f"  {alpha:6.1f}  {r['mean_auc']:8.4f}  {diff:+8.4f}")

# Find overall best configuration
all_configs = []
for n_aug, r in results_n_aug.items():
    all_configs.append(("n_aug", n_aug, 0.2, r["mean_auc"]))
for alpha, r in results_alpha.items():
    all_configs.append(("alpha", 100, alpha, r["mean_auc"]))

best_config = max(all_configs, key=lambda x: x[3])
overall_best_auc = best_config[3]


# ============================================================
# Phase 5: Generate Submission
# ============================================================
print("\n" + "=" * 60)
print("Phase 5: Submission Generation")
print("=" * 60)

sample_submit = pd.read_csv(SAMPLE_SUBMIT_CSV, index_col=0, header=None)

if overall_best_auc > baseline_auc:
    final_n_aug = best_config[1] if best_config[0] == "n_aug" else 100
    final_alpha = best_config[2] if best_config[0] == "alpha" else 0.2
    print(f"  Best config: n_aug={final_n_aug}, alpha={final_alpha}")
    print(f"  AUC improvement: {overall_best_auc - baseline_auc:+.4f}")
    print("  Generating augmented submission...")

    X_tr_full, X_te_full = fb(X_raw, test)

    X_aug_final, y_aug_final = mixup_augment(
        X_tr_full, y, n_aug=final_n_aug, alpha=final_alpha, seed=SEED,
    )
    X_train_aug = pd.concat([X_tr_full, X_aug_final], ignore_index=True)
    y_train_aug = pd.concat([y, y_aug_final], ignore_index=True)

    print(f"  Final training size: {len(X_train_aug)} "
          f"({len(X_tr_full)} + {len(X_aug_final)})")

    model = make_tuned_voting()
    model.fit(X_train_aug, y_train_aug)
    test_pred = model.predict_proba(X_te_full)[:, 1]

    submission = sample_submit.copy()
    submission[1] = test_pred
    submission.to_csv("submit_augmentation.csv", header=None)
    print("  Saved: submit_augmentation.csv")
else:
    print("  Augmentation did NOT improve over baseline.")
    print("  Generating baseline submission...")

    X_tr_full, X_te_full = fb(X_raw, test)
    model = make_tuned_voting()
    model.fit(X_tr_full, y)
    test_pred = model.predict_proba(X_te_full)[:, 1]

    submission = sample_submit.copy()
    submission[1] = test_pred
    submission.to_csv("submit_augmentation_baseline.csv", header=None)
    print("  Saved: submit_augmentation_baseline.csv")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

print(f"  Baseline Voting CV AUC: {baseline_auc:.4f}")
print(f"  Best augmented AUC:     {overall_best_auc:.4f}")
print(f"  Best config:            n_aug={best_config[1]}, alpha={best_config[2]}")
print(f"  Improvement:            {overall_best_auc - baseline_auc:+.4f}")

if overall_best_auc > baseline_auc:
    print("\n  Mixup augmentation IMPROVED performance.")
    print("  This suggests the model benefits from additional interpolated samples.")
else:
    print("\n  Mixup augmentation did NOT improve performance.")
    print("  With 445 samples, tree-based models may already handle")
    print("  the feature space well via bagging/boosting internal augmentation.")

print("\nDone!")
