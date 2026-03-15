"""Experiment 25: Advanced Stacking (Nested CV).

Hypothesis: Stacking with tuned models and domain+missing features
can outperform Equal Voting by learning optimal blending weights.

Phase 1: sklearn StackingClassifier with LogReg meta-learner
Phase 2: Test different meta-learner configurations (C values, passthrough)
Phase 3: Manual nested CV for fine-grained control
Phase 4: Compare stacking vs Equal Voting
Phase 5: Generate submission if improved
"""
import sys
sys.path.append("../..")

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import TRAIN_CSV, TEST_CSV, SAMPLE_SUBMIT_CSV, SEED, N_FOLDS, TARGET_COL
from src.utils import seed_everything
from src.exp_features import make_exp_builder
from src.evaluation import cross_validate, cross_validate_oof, get_cv_splitter, calc_metrics

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


MODEL_FNS = {
    "LogReg": make_tuned_logreg,
    "RF": make_tuned_rf,
    "XGB": make_tuned_xgb,
    "LGBM": make_tuned_lgbm,
}

# ============================================================
# Baseline: Equal Voting
# ============================================================
print("=" * 60)
print("Baseline: Equal Voting")
print("=" * 60)

_, voting_mean_m = cross_validate(make_tuned_voting, X_raw, y, feature_builder=fb)
print(f"  Voting: AUC={voting_mean_m['auc']:.4f}, Acc={voting_mean_m['accuracy']:.4f}")

# Individual model baselines
base_results = {}
for name, fn in MODEL_FNS.items():
    _, mean_m = cross_validate(fn, X_raw, y, feature_builder=fb)
    base_results[name] = mean_m
    print(f"  {name}: AUC={mean_m['auc']:.4f}")


# ============================================================
# Phase 1: sklearn StackingClassifier
# ============================================================
print("\n" + "=" * 60)
print("Phase 1: StackingClassifier (default meta_C=0.1)")
print("=" * 60)


def make_stacking(meta_C=0.1, passthrough=False):
    """Create a StackingClassifier with LogReg meta-learner."""
    return StackingClassifier(
        estimators=[
            ("logreg", make_tuned_logreg()),
            ("rf", make_tuned_rf()),
            ("xgb", make_tuned_xgb()),
            ("lgbm", make_tuned_lgbm()),
        ],
        final_estimator=LogisticRegression(
            C=meta_C, random_state=SEED, max_iter=2000,
        ),
        cv=5,
        passthrough=passthrough,
    )


# Default stacking
_, stack_mean = cross_validate(
    lambda: make_stacking(meta_C=0.1, passthrough=False),
    X_raw, y, feature_builder=fb,
)
print(f"  Stacking (C=0.1, no passthrough): AUC={stack_mean['auc']:.4f}")
diff = stack_mean["auc"] - voting_mean_m["auc"]
print(f"  vs Voting: {diff:+.4f}")


# ============================================================
# Phase 2: Meta-Learner Configuration Search
# ============================================================
print("\n" + "=" * 60)
print("Phase 2: Meta-Learner Configuration Search")
print("=" * 60)

C_VALUES = [0.01, 0.1, 1.0, 10.0]
PASSTHROUGH_OPTIONS = [False, True]

stack_results = {}

print(f"\n  {'Config':<40} {'AUC':>8} {'vs Voting':>10}")
print("  " + "-" * 62)

for passthrough in PASSTHROUGH_OPTIONS:
    for meta_C in C_VALUES:
        config_name = f"C={meta_C}, pass={'Yes' if passthrough else 'No'}"

        _, mean_m = cross_validate(
            lambda c=meta_C, p=passthrough: make_stacking(meta_C=c, passthrough=p),
            X_raw, y, feature_builder=fb,
        )
        stack_results[config_name] = mean_m
        diff = mean_m["auc"] - voting_mean_m["auc"]
        marker = " ***" if diff > 0.001 else ""
        print(f"  {config_name:<40} {mean_m['auc']:>8.4f} {diff:>+10.4f}{marker}")

# Find best stacking config
best_config = max(stack_results, key=lambda k: stack_results[k]["auc"])
best_stack_auc = stack_results[best_config]["auc"]
print(f"\n  Best stacking config: {best_config} (AUC={best_stack_auc:.4f})")


# ============================================================
# Phase 3: Manual Nested CV
# ============================================================
print("\n" + "=" * 60)
print("Phase 3: Manual Nested CV (Outer 5-fold, Inner OOF)")
print("=" * 60)

outer_cv = get_cv_splitter(n_folds=N_FOLDS, seed=SEED)
outer_aucs = []
outer_accs = []
all_outer_preds = np.zeros(len(y))
all_outer_true = np.zeros(len(y))

META_C_VALUES = [0.01, 0.1, 1.0, 10.0]
nested_results_by_C = {c: {"aucs": [], "preds": np.zeros(len(y))} for c in META_C_VALUES}

for fold, (outer_tr_idx, outer_te_idx) in enumerate(outer_cv.split(X_raw, y)):
    X_outer_tr = X_raw.iloc[outer_tr_idx]
    X_outer_te = X_raw.iloc[outer_te_idx]
    y_outer_tr = y.iloc[outer_tr_idx]
    y_outer_te = y.iloc[outer_te_idx]

    print(f"\n  Outer Fold {fold}: train={len(outer_tr_idx)}, test={len(outer_te_idx)}")

    # Inner OOF predictions for meta-features
    n_models = len(MODEL_FNS)
    oof_meta = np.zeros((len(outer_tr_idx), n_models))
    test_meta = np.zeros((len(outer_te_idx), n_models))

    for i, (name, model_fn) in enumerate(MODEL_FNS.items()):
        # Inner CV on outer-train to get OOF predictions
        inner_fb = make_exp_builder(
            missing_flags=True, age_bins=None, fare_bins=None,
            interactions=False, polynomial=False, group_stats=False,
            freq_encoding=False, rank_features=False, domain_features=True,
        )
        _, _, _, oof_proba, _ = cross_validate_oof(
            model_fn, X_outer_tr, y_outer_tr,
            n_folds=N_FOLDS, seed=SEED, feature_builder=inner_fb,
        )
        oof_meta[:, i] = oof_proba

        # Train on full outer-train, predict on outer-test
        X_tr_full, X_te_full = inner_fb(X_outer_tr, X_outer_te)
        model = model_fn()
        model.fit(X_tr_full, y_outer_tr)
        test_meta[:, i] = model.predict_proba(X_te_full)[:, 1]

        print(f"    {name}: inner OOF done, test predicted")

    # Try different meta-learner C values
    for meta_C in META_C_VALUES:
        meta = LogisticRegression(C=meta_C, random_state=SEED, max_iter=2000)
        meta.fit(oof_meta, y_outer_tr)
        meta_pred = meta.predict_proba(test_meta)[:, 1]
        fold_auc = roc_auc_score(y_outer_te, meta_pred)
        nested_results_by_C[meta_C]["aucs"].append(fold_auc)
        nested_results_by_C[meta_C]["preds"][outer_te_idx] = meta_pred

    # Default C=0.1 for main tracking
    meta = LogisticRegression(C=0.1, random_state=SEED, max_iter=2000)
    meta.fit(oof_meta, y_outer_tr)
    meta_pred = meta.predict_proba(test_meta)[:, 1]
    meta_class = (meta_pred >= 0.5).astype(int)

    fold_auc = roc_auc_score(y_outer_te, meta_pred)
    outer_aucs.append(fold_auc)
    all_outer_preds[outer_te_idx] = meta_pred

    # Also report meta-learner weights
    print(f"    Meta weights (C=0.1): {dict(zip(MODEL_FNS.keys(), meta.coef_[0].round(3)))}")
    print(f"    Fold {fold} AUC: {fold_auc:.4f}")

# Manual nested CV summary
print(f"\n  Manual Nested CV (C=0.1):")
print(f"    Mean AUC: {np.mean(outer_aucs):.4f} +/- {np.std(outer_aucs):.4f}")
print(f"    Per-fold: {[f'{a:.4f}' for a in outer_aucs]}")

overall_nested_auc = roc_auc_score(y, all_outer_preds)
print(f"    Overall OOF AUC: {overall_nested_auc:.4f}")

# Compare across C values
print(f"\n  Nested CV by Meta-Learner C:")
print(f"  {'C':>8} {'Mean AUC':>10} {'Std':>8} {'OOF AUC':>10}")
print("  " + "-" * 40)
best_nested_C = None
best_nested_auc = -1
for meta_C in META_C_VALUES:
    aucs = nested_results_by_C[meta_C]["aucs"]
    oof_auc = roc_auc_score(y, nested_results_by_C[meta_C]["preds"])
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"  {meta_C:>8.2f} {mean_auc:>10.4f} {std_auc:>8.4f} {oof_auc:>10.4f}")
    if mean_auc > best_nested_auc:
        best_nested_auc = mean_auc
        best_nested_C = meta_C

print(f"\n  Best nested CV meta C: {best_nested_C} (Mean AUC={best_nested_auc:.4f})")


# ============================================================
# Phase 4: Compare Stacking vs Equal Voting
# ============================================================
print("\n" + "=" * 60)
print("Phase 4: Stacking vs Equal Voting Comparison")
print("=" * 60)

print(f"\n  {'Method':<45} {'AUC':>8} {'vs Voting':>10}")
print("  " + "-" * 67)

# Voting baseline
print(f"  {'Equal Voting (baseline)':<45} {voting_mean_m['auc']:>8.4f} {'---':>10}")

# Best sklearn stacking
diff = best_stack_auc - voting_mean_m["auc"]
print(f"  {'sklearn Stacking (' + best_config + ')':<45} {best_stack_auc:>8.4f} {diff:>+10.4f}")

# Manual nested CV (best C)
diff = best_nested_auc - voting_mean_m["auc"]
print(f"  {'Manual Nested CV (C=' + str(best_nested_C) + ')':<45} {best_nested_auc:>8.4f} {diff:>+10.4f}")

# Determine overall best
all_methods = {
    "voting": voting_mean_m["auc"],
    "sklearn_stacking": best_stack_auc,
    "nested_cv": best_nested_auc,
}
best_method = max(all_methods, key=all_methods.get)
print(f"\n  Best overall method: {best_method} (AUC={all_methods[best_method]:.4f})")


# ============================================================
# Phase 5: Generate Submission if Improved
# ============================================================
print("\n" + "=" * 60)
print("Phase 5: Submission Generation")
print("=" * 60)

sample_submit = pd.read_csv(SAMPLE_SUBMIT_CSV, index_col=0, header=None)

# Build features for full training and test data
X_tr_full, X_te_full = fb(X_raw, test)
print(f"  Full train features: {X_tr_full.shape}")
print(f"  Full test features:  {X_te_full.shape}")

# --- Submission 1: Manual stacking (full OOF -> meta -> predict test) ---
# Step 1: Get full OOF predictions on all training data
oof_full = np.zeros((len(y), len(MODEL_FNS)))
test_preds_all = np.zeros((len(test), len(MODEL_FNS)))

inner_fb = make_exp_builder(
    missing_flags=True, age_bins=None, fare_bins=None,
    interactions=False, polynomial=False, group_stats=False,
    freq_encoding=False, rank_features=False, domain_features=True,
)

for i, (name, model_fn) in enumerate(MODEL_FNS.items()):
    # Full OOF on training data
    _, _, _, oof_proba, _ = cross_validate_oof(
        model_fn, X_raw, y, n_folds=N_FOLDS, seed=SEED, feature_builder=inner_fb,
    )
    oof_full[:, i] = oof_proba

    # Train on all training data, predict test
    model = model_fn()
    model.fit(X_tr_full, y)
    test_preds_all[:, i] = model.predict_proba(X_te_full)[:, 1]
    print(f"  {name}: OOF and test predictions done")

# Step 2: Train meta-learner on full OOF
meta = LogisticRegression(C=best_nested_C, random_state=SEED, max_iter=2000)
meta.fit(oof_full, y)
print(f"\n  Meta-learner weights (C={best_nested_C}):")
for name, w in zip(MODEL_FNS.keys(), meta.coef_[0]):
    print(f"    {name}: {w:.4f}")
print(f"    intercept: {meta.intercept_[0]:.4f}")

# Step 3: Predict test with meta-learner
stacking_pred = meta.predict_proba(test_preds_all)[:, 1]
sub_stacking = sample_submit.copy()
sub_stacking[1] = stacking_pred
sub_stacking.to_csv("submit_stacking.csv", header=None)
print(f"\n  submit_stacking.csv: mean={stacking_pred.mean():.3f}, std={stacking_pred.std():.3f}")

# --- Submission 2: Voting baseline ---
voting_model = make_tuned_voting()
voting_model.fit(X_tr_full, y)
voting_pred = voting_model.predict_proba(X_te_full)[:, 1]
sub_voting = sample_submit.copy()
sub_voting[1] = voting_pred
sub_voting.to_csv("submit_voting_baseline.csv", header=None)
print(f"  submit_voting_baseline.csv: mean={voting_pred.mean():.3f}, std={voting_pred.std():.3f}")

# --- Submission 3: sklearn StackingClassifier with best config ---
# Parse best config to extract C and passthrough
best_C_str = best_config.split("C=")[1].split(",")[0]
best_C_val = float(best_C_str)
best_pass = "Yes" in best_config

sklearn_stack = make_stacking(meta_C=best_C_val, passthrough=best_pass)
sklearn_stack.fit(X_tr_full, y)
sklearn_pred = sklearn_stack.predict_proba(X_te_full)[:, 1]
sub_sklearn = sample_submit.copy()
sub_sklearn[1] = sklearn_pred
sub_sklearn.to_csv("submit_sklearn_stacking.csv", header=None)
print(f"  submit_sklearn_stacking.csv: mean={sklearn_pred.mean():.3f}, std={sklearn_pred.std():.3f}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

improved = all_methods[best_method] > voting_mean_m["auc"]

if improved:
    gain = all_methods[best_method] - voting_mean_m["auc"]
    print(f"\n  Stacking IMPROVED over Voting!")
    print(f"  Best method: {best_method} (AUC={all_methods[best_method]:.4f}, +{gain:.4f})")
    if best_method == "nested_cv":
        print(f"  Recommended submission: submit_stacking.csv")
    else:
        print(f"  Recommended submission: submit_sklearn_stacking.csv")
else:
    print(f"\n  Stacking did NOT improve over Voting (AUC={voting_mean_m['auc']:.4f})")
    print(f"  Recommended submission: submit_voting_baseline.csv")

print(f"\n  All results:")
for method, auc in sorted(all_methods.items(), key=lambda x: -x[1]):
    diff = auc - voting_mean_m["auc"]
    print(f"    {method:<25} AUC={auc:.4f} ({diff:+.4f} vs voting)")

print("\nDone!")
