"""Experiment 34: Bayesian Model Averaging.

Hypothesis: Logistic regression with different feature subsets and regularization
strengths, averaged by posterior probability (approximated via softmax of CV AUC),
provides natural regularization and diverse prediction sources.

Phase 1: Define informative feature subsets
Phase 2: OOF predictions for each (subset, C) combination
Phase 3: Bayesian Model Averaging with softmax-weighted predictions
Phase 4: Combine BMA with tree-based models
Phase 5: Compare with baseline Equal Voting
Phase 6: Generate submission if improved
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

C_VALUES = [0.001, 0.01, 0.1, 1.0]
SOFTMAX_TEMPERATURE = 50.0

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
# Phase 0: Baseline CV
# ============================================================
print("=" * 60)
print("Experiment 34: Bayesian Model Averaging")
print("=" * 60)

print("\n--- Phase 0: Baseline CV (Equal Voting) ---")
_, baseline_mean = cross_validate(
    make_tuned_voting, X_raw, y, n_folds=N_FOLDS, seed=SEED, feature_builder=fb,
)
baseline_auc = baseline_mean["auc"]
print(f"  Baseline Voting AUC: {baseline_auc:.4f}")


# ============================================================
# Phase 1: Define feature subsets
# ============================================================
print("\n--- Phase 1: Feature Subsets ---")

# Build features once to inspect available columns
X_demo, _ = fb(X_raw, test)
all_features = list(X_demo.columns)
print(f"  Total available features ({len(all_features)}): {all_features}")

# Define subsets using available features (filter to only existing ones)
FEATURE_SUBSETS = {
    "demographics": ["sex", "age", "pclass", "is_child", "is_alone"],
    "economic": ["fare", "pclass", "log_fare", "fare_per_person", "fare_zero"],
    "family": [
        "sibsp", "parch", "family_size", "is_alone",
        "family_small", "family_large", "is_mother",
    ],
    "full_basic": ["sex", "age", "pclass", "sibsp", "parch", "fare"],
    "domain_strong": None,  # Use all features (domain+missing)
}

# Show which features are actually available per subset
for subset_name, feat_list in FEATURE_SUBSETS.items():
    if feat_list is None:
        print(f"  {subset_name}: ALL ({len(all_features)} features)")
    else:
        available = [f for f in feat_list if f in all_features]
        missing = [f for f in feat_list if f not in all_features]
        print(f"  {subset_name}: {len(available)} available, "
              f"{len(missing)} missing {missing if missing else ''}")


# ============================================================
# Phase 2: OOF predictions for each (subset, C) combination
# ============================================================
print("\n--- Phase 2: OOF Predictions per (Subset, C) ---")

cv = get_cv_splitter(n_folds=N_FOLDS, seed=SEED)

all_oof_probas = []
all_test_probas = []
all_weights = []
all_labels = []

for subset_name, feat_list in FEATURE_SUBSETS.items():
    for C in C_VALUES:
        oof_proba = np.zeros(len(y))
        test_proba_folds = []
        fold_aucs = []

        cv = get_cv_splitter(n_folds=N_FOLDS, seed=SEED)

        for fold, (tr_idx, va_idx) in enumerate(cv.split(X_raw, y)):
            X_tr_raw = X_raw.iloc[tr_idx]
            X_va_raw = X_raw.iloc[va_idx]
            y_tr = y.iloc[tr_idx]
            y_va = y.iloc[va_idx]

            # Build features
            X_tr, X_va = fb(X_tr_raw, X_va_raw)

            # Select feature subset
            if feat_list is not None:
                available = [f for f in feat_list if f in X_tr.columns]
                X_tr_sub = X_tr[available]
                X_va_sub = X_va[available]
            else:
                X_tr_sub = X_tr
                X_va_sub = X_va

            # Scale and fit LogReg
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_sub)
            X_va_s = scaler.transform(X_va_sub)

            model = LogisticRegression(
                C=C, max_iter=2000, random_state=SEED, solver="liblinear",
            )
            model.fit(X_tr_s, y_tr)
            oof_proba[va_idx] = model.predict_proba(X_va_s)[:, 1]
            fold_aucs.append(roc_auc_score(y_va, oof_proba[va_idx]))

        mean_auc = np.mean(fold_aucs)
        all_oof_probas.append(oof_proba)
        all_weights.append(mean_auc)
        all_labels.append(f"{subset_name}(C={C})")

        print(f"  {subset_name:15s} (C={C:6.3f}): AUC={mean_auc:.4f}  "
              f"folds={[f'{a:.4f}' for a in fold_aucs]}")


# ============================================================
# Phase 3: Bayesian Model Averaging
# ============================================================
print("\n--- Phase 3: Bayesian Model Averaging ---")

# Softmax weighting by CV AUC
weights = np.array(all_weights)
print(f"  Raw AUC range: [{weights.min():.4f}, {weights.max():.4f}]")

# Temperature-scaled softmax
weights_exp = np.exp(SOFTMAX_TEMPERATURE * (weights - weights.max()))
weights_norm = weights_exp / weights_exp.sum()

print(f"\n  Model weights (top 10 by weight):")
sorted_idx = np.argsort(weights_norm)[::-1]
for rank, idx in enumerate(sorted_idx[:10]):
    print(f"    #{rank+1}: {all_labels[idx]:30s}  "
          f"AUC={all_weights[idx]:.4f}  weight={weights_norm[idx]:.4f}")

# Equal-weighted BMA
bma_proba_equal = np.mean(all_oof_probas, axis=0)
bma_auc_equal = roc_auc_score(y, bma_proba_equal)
print(f"\n  Equal-weighted BMA OOF AUC: {bma_auc_equal:.4f}")

# Softmax-weighted BMA
bma_proba_weighted = np.average(all_oof_probas, axis=0, weights=weights_norm)
bma_auc_weighted = roc_auc_score(y, bma_proba_weighted)
print(f"  Softmax-weighted BMA OOF AUC: {bma_auc_weighted:.4f}")

# Top-K averaging (use only top K models by weight)
for top_k in [3, 5, 10]:
    top_indices = sorted_idx[:top_k]
    top_probas = [all_oof_probas[i] for i in top_indices]
    top_weights = weights_norm[top_indices]
    top_weights_renorm = top_weights / top_weights.sum()

    bma_topk = np.average(top_probas, axis=0, weights=top_weights_renorm)
    bma_topk_auc = roc_auc_score(y, bma_topk)
    print(f"  Top-{top_k} weighted BMA OOF AUC: {bma_topk_auc:.4f}")


# ============================================================
# Phase 4: Combine BMA with Tree-Based Models
# ============================================================
print("\n--- Phase 4: BMA + Tree-Based Ensemble ---")

# Get OOF predictions from tree-based models
tree_models = {
    "RF": make_tuned_rf,
    "XGB": make_tuned_xgb,
    "LGBM": make_tuned_lgbm,
}

tree_oof = {}
for name, model_fn in tree_models.items():
    oof_proba = np.zeros(len(y))
    cv = get_cv_splitter(n_folds=N_FOLDS, seed=SEED)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_raw, y)):
        X_tr_raw = X_raw.iloc[tr_idx]
        X_va_raw = X_raw.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        X_tr, X_va = fb(X_tr_raw, X_va_raw)

        model = model_fn()
        model.fit(X_tr, y_tr)
        oof_proba[va_idx] = model.predict_proba(X_va)[:, 1]

    tree_auc = roc_auc_score(y, oof_proba)
    tree_oof[name] = oof_proba
    print(f"  {name} OOF AUC: {tree_auc:.4f}")

# Combine BMA + trees at different blend ratios
print("\n  BMA + Trees blending:")
bma_best = bma_proba_weighted  # Use softmax-weighted BMA
tree_avg = np.mean(list(tree_oof.values()), axis=0)
tree_avg_auc = roc_auc_score(y, tree_avg)
print(f"  Tree average OOF AUC: {tree_avg_auc:.4f}")

blend_results = {}
for bma_weight in [0.1, 0.2, 0.3, 0.4, 0.5]:
    blended = bma_weight * bma_best + (1 - bma_weight) * tree_avg
    blended_auc = roc_auc_score(y, blended)
    blend_results[bma_weight] = blended_auc
    print(f"  BMA_w={bma_weight:.1f}: AUC={blended_auc:.4f}  "
          f"diff_vs_baseline={blended_auc - baseline_auc:+.4f}")


# ============================================================
# Phase 5: Comparison
# ============================================================
print("\n" + "=" * 60)
print("Phase 5: Comparison Summary")
print("=" * 60)

print(f"\n  Baseline Equal Voting AUC:    {baseline_auc:.4f}")
print(f"  BMA Equal-weighted AUC:       {bma_auc_equal:.4f}  "
      f"({bma_auc_equal - baseline_auc:+.4f})")
print(f"  BMA Softmax-weighted AUC:     {bma_auc_weighted:.4f}  "
      f"({bma_auc_weighted - baseline_auc:+.4f})")
print(f"  Tree Average AUC:             {tree_avg_auc:.4f}  "
      f"({tree_avg_auc - baseline_auc:+.4f})")

best_blend_w = max(blend_results, key=blend_results.get)
best_blend_auc = blend_results[best_blend_w]
print(f"  Best BMA+Trees Blend AUC:     {best_blend_auc:.4f}  "
      f"(BMA_w={best_blend_w:.1f}, "
      f"diff={best_blend_auc - baseline_auc:+.4f})")

# Find overall best approach
approaches = {
    "BMA Equal": bma_auc_equal,
    "BMA Softmax": bma_auc_weighted,
    f"BMA+Trees (w={best_blend_w})": best_blend_auc,
}
best_approach_name = max(approaches, key=approaches.get)
best_approach_auc = approaches[best_approach_name]


# ============================================================
# Phase 6: Generate Submission
# ============================================================
print("\n" + "=" * 60)
print("Phase 6: Submission Generation")
print("=" * 60)

sample_submit = pd.read_csv(SAMPLE_SUBMIT_CSV, index_col=0, header=None)

# Always generate the BMA submission for comparison
# Train all (subset, C) combinations on full data, predict test
print("\n  Training all BMA models on full data...")

X_tr_full, X_te_full = fb(X_raw, test)

all_test_preds = []
for subset_name, feat_list in FEATURE_SUBSETS.items():
    for C in C_VALUES:
        if feat_list is not None:
            available = [f for f in feat_list if f in X_tr_full.columns]
            X_tr_sub = X_tr_full[available]
            X_te_sub = X_te_full[available]
        else:
            X_tr_sub = X_tr_full
            X_te_sub = X_te_full

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_sub)
        X_te_s = scaler.transform(X_te_sub)

        model = LogisticRegression(
            C=C, max_iter=2000, random_state=SEED, solver="liblinear",
        )
        model.fit(X_tr_s, y)
        test_pred = model.predict_proba(X_te_s)[:, 1]
        all_test_preds.append(test_pred)

# Softmax-weighted BMA test predictions
bma_test_pred = np.average(all_test_preds, axis=0, weights=weights_norm)

if best_approach_auc > baseline_auc:
    print(f"\n  Best approach: {best_approach_name} (AUC={best_approach_auc:.4f})")

    if "Blend" in best_approach_name:
        # Need tree predictions on test too
        tree_test_preds = []
        for name, model_fn in tree_models.items():
            model = model_fn()
            model.fit(X_tr_full, y)
            pred = model.predict_proba(X_te_full)[:, 1]
            tree_test_preds.append(pred)
            print(f"  {name} test pred: mean={pred.mean():.4f}")

        tree_test_avg = np.mean(tree_test_preds, axis=0)
        final_pred = best_blend_w * bma_test_pred + (1 - best_blend_w) * tree_test_avg
        filename = "submit_bma_blend.csv"
    else:
        final_pred = bma_test_pred
        filename = "submit_bma.csv"

    submission = sample_submit.copy()
    submission[1] = final_pred
    submission.to_csv(filename, header=None)
    print(f"  Saved: {filename}")
else:
    print("\n  BMA did NOT improve over baseline Equal Voting.")
    print("  Generating BMA submission anyway for comparison...")

    submission = sample_submit.copy()
    submission[1] = bma_test_pred
    submission.to_csv("submit_bma_reference.csv", header=None)
    print("  Saved: submit_bma_reference.csv")

    # Also save baseline voting submission
    model = make_tuned_voting()
    model.fit(X_tr_full, y)
    baseline_pred = model.predict_proba(X_te_full)[:, 1]

    submission_baseline = sample_submit.copy()
    submission_baseline[1] = baseline_pred
    submission_baseline.to_csv("submit_bma_baseline.csv", header=None)
    print("  Saved: submit_bma_baseline.csv")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

print(f"  Baseline Equal Voting CV AUC: {baseline_auc:.4f}")
print(f"  Best BMA approach:            {best_approach_name}")
print(f"  Best BMA AUC:                 {best_approach_auc:.4f}")
print(f"  Improvement:                  {best_approach_auc - baseline_auc:+.4f}")
print(f"\n  Feature subsets tested:        {len(FEATURE_SUBSETS)}")
print(f"  C values tested:              {C_VALUES}")
print(f"  Total (subset, C) combos:     {len(all_oof_probas)}")
print(f"  Softmax temperature:          {SOFTMAX_TEMPERATURE}")

if best_approach_auc > baseline_auc:
    print("\n  Bayesian Model Averaging IMPROVED performance.")
    print("  The diversity from feature subsets provides complementary signals.")
else:
    print("\n  Bayesian Model Averaging did NOT improve over Equal Voting.")
    print("  The tree-based ensemble already captures sufficient diversity.")

print("\nDone!")
