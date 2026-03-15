"""Experiment 35: Probability Blend Optimization.

Three-party consensus strategy: blend Voting, Stacking, and Rank ensemble
predictions at the probability level for AUC improvement.

Strategy 1: Linear blend  p = a*vote + b*stack + c*rank  (a+b+c=1)
Strategy 2: Gated blend   (disagreement-based routing between Stacking & Rank)

Phases:
  1. Generate base model OOF predictions (4 models)
  2. Derive ensemble-level OOFs (Voting, Stacking, Rank)
  3. Linear blend optimization (grid search + scipy.optimize)
  4. Gated/Selective blend (disagreement-based routing)
  5. Repeated CV stability validation (5-fold x 5-seed)
  6. Generate test predictions and submission files
  7. Evaluate against local ground truth
"""
import sys
sys.path.append("../..")

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import (TRAIN_CSV, TEST_CSV, SAMPLE_SUBMIT_CSV, SEED,
                        N_FOLDS, TARGET_COL)
from src.utils import seed_everything
from src.exp_features import make_exp_builder
from src.evaluation import cross_validate_oof, evaluate_submission

seed_everything(SEED)

# ============================================================
# Data Loading
# ============================================================
train = pd.read_csv(TRAIN_CSV, index_col=0)
test = pd.read_csv(TEST_CSV, index_col=0)
sample_submit = pd.read_csv(SAMPLE_SUBMIT_CSV, index_col=0, header=None)

y = train[TARGET_COL]
X_raw = train.drop(columns=[TARGET_COL])

print(f"Train: {X_raw.shape}, Test: {test.shape}")

# ============================================================
# Configuration: BEST_PARAMS + Feature Builder
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

fb = make_exp_builder(
    missing_flags=True, age_bins=None, fare_bins=None,
    interactions=False, polynomial=False, group_stats=False,
    freq_encoding=False, rank_features=False, domain_features=True,
)


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
    return XGBClassifier(**p, random_state=SEED, eval_metric="logloss")


def make_tuned_lgbm():
    p = BEST_PARAMS["LightGBM"]
    return LGBMClassifier(**p, random_state=SEED, verbose=-1)


MODEL_FNS = {
    "LogReg": make_tuned_logreg,
    "RF": make_tuned_rf,
    "XGB": make_tuned_xgb,
    "LGBM": make_tuned_lgbm,
}


# ============================================================
# Phase 1: Base Model OOF Predictions
# ============================================================
print("=" * 60)
print("Phase 1: Base Model OOF Predictions")
print("=" * 60)

base_oof = {}
base_models = {}

for name, fn in MODEL_FNS.items():
    _, mean_m, _, oof_proba, models = cross_validate_oof(
        fn, X_raw, y, feature_builder=fb,
    )
    base_oof[name] = oof_proba
    base_models[name] = models
    print(f"  {name}: OOF AUC={mean_m['auc']:.4f}, Acc={mean_m['accuracy']:.4f}")

# Base OOF array (445 x 4)
oof_arr = np.column_stack([base_oof[name] for name in MODEL_FNS])
model_names = list(MODEL_FNS.keys())

# Pairwise correlations
print(f"\n  OOF Probability Correlations:")
corr = pd.DataFrame(oof_arr, columns=model_names).corr()
for i, n1 in enumerate(model_names):
    for j, n2 in enumerate(model_names):
        if j > i:
            print(f"    {n1}-{n2}: {corr.iloc[i, j]:.3f}")


# ============================================================
# Phase 2: Ensemble-Level OOFs (Voting, Stacking, Rank)
# ============================================================
print("\n" + "=" * 60)
print("Phase 2: Ensemble-Level OOF Predictions")
print("=" * 60)

# --- Voting OOF: simple mean of 4 model probabilities ---
voting_oof = oof_arr.mean(axis=1)
voting_auc = roc_auc_score(y, voting_oof)
print(f"  Voting OOF AUC:   {voting_auc:.4f}")

# --- Rank OOF: rank-normalize each model, then average ---
rank_arr = np.column_stack([
    rankdata(base_oof[name]) / len(y) for name in MODEL_FNS
])
rank_oof = rank_arr.mean(axis=1)
rank_auc = roc_auc_score(y, rank_oof)
print(f"  Rank OOF AUC:     {rank_auc:.4f}")

# --- Stacking OOF: LogReg meta-learner with CV on base OOFs ---
# Use different seed from base models to reduce leakage correlation
META_SEEDS = [123, 456, 789]
META_C_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0]

print(f"\n  Stacking Meta-Learner Search:")
print(f"  {'C':>8} {'Mean AUC':>10} {'Std':>8}")
print("  " + "-" * 30)

best_meta_C = 0.1
best_meta_auc = -1

for meta_C in META_C_VALUES:
    seed_aucs = []
    for ms in META_SEEDS:
        meta_oof = np.zeros(len(y))
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=ms)
        for _, (tr_idx, va_idx) in enumerate(cv.split(oof_arr, y)):
            meta = LogisticRegression(C=meta_C, random_state=SEED, max_iter=2000)
            meta.fit(oof_arr[tr_idx], y.values[tr_idx])
            meta_oof[va_idx] = meta.predict_proba(oof_arr[va_idx])[:, 1]
        seed_aucs.append(roc_auc_score(y, meta_oof))

    mean_auc = np.mean(seed_aucs)
    std_auc = np.std(seed_aucs)
    marker = " ***" if mean_auc > best_meta_auc else ""
    print(f"  {meta_C:>8.2f} {mean_auc:>10.4f} {std_auc:>8.4f}{marker}")

    if mean_auc > best_meta_auc:
        best_meta_auc = mean_auc
        best_meta_C = meta_C

print(f"\n  Best meta C: {best_meta_C} (AUC={best_meta_auc:.4f})")

# Generate final Stacking OOF with best C (average over seeds for stability)
stacking_oof = np.zeros(len(y))
for ms in META_SEEDS:
    temp_oof = np.zeros(len(y))
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=ms)
    for _, (tr_idx, va_idx) in enumerate(cv.split(oof_arr, y)):
        meta = LogisticRegression(C=best_meta_C, random_state=SEED, max_iter=2000)
        meta.fit(oof_arr[tr_idx], y.values[tr_idx])
        temp_oof[va_idx] = meta.predict_proba(oof_arr[va_idx])[:, 1]
    stacking_oof += temp_oof
stacking_oof /= len(META_SEEDS)

stacking_auc = roc_auc_score(y, stacking_oof)
print(f"  Stacking OOF AUC: {stacking_auc:.4f} (avg over {len(META_SEEDS)} seeds)")

# Summary of 3 ensemble-level OOFs
print(f"\n  Ensemble OOF Summary:")
print(f"    Voting:   AUC={voting_auc:.4f}")
print(f"    Stacking: AUC={stacking_auc:.4f}")
print(f"    Rank:     AUC={rank_auc:.4f}")

# Correlation between ensemble OOFs
ens_corr = np.corrcoef(np.vstack([voting_oof, stacking_oof, rank_oof]))
print(f"\n  Ensemble-Level Correlations:")
ens_names = ["Voting", "Stacking", "Rank"]
for i in range(3):
    for j in range(i + 1, 3):
        print(f"    {ens_names[i]}-{ens_names[j]}: {ens_corr[i, j]:.4f}")


# ============================================================
# Phase 3: Linear Blend Optimization
# ============================================================
print("\n" + "=" * 60)
print("Phase 3: Linear Blend Optimization")
print("=" * 60)
print("  p_final = a * voting + b * stacking + c * rank  (a+b+c=1)")


def blend_auc(weights, p_voting, p_stacking, p_rank, y_true):
    """Compute AUC for a given weight vector."""
    a, b, c = weights
    p_blend = a * p_voting + b * p_stacking + c * p_rank
    return roc_auc_score(y_true, p_blend)


# --- Grid Search over simplex (step=0.05) ---
print("\n  Grid Search (step=0.05):")
best_grid_weights = None
best_grid_auc = -1

step = 0.05
grid_results = []

for a_int in range(0, 21):
    a = a_int * step
    for b_int in range(0, 21 - a_int):
        b = b_int * step
        c = 1.0 - a - b
        if c < -1e-9:
            continue
        c = max(c, 0.0)

        auc = blend_auc([a, b, c], voting_oof, stacking_oof, rank_oof, y)
        grid_results.append((a, b, c, auc))

        if auc > best_grid_auc:
            best_grid_auc = auc
            best_grid_weights = (a, b, c)

# Top 10 grid results
grid_results.sort(key=lambda x: -x[3])
print(f"\n  Top 10 Weight Combinations:")
print(f"  {'a (Vote)':>10} {'b (Stack)':>10} {'c (Rank)':>10} {'AUC':>10}")
print("  " + "-" * 44)
for a, b, c, auc in grid_results[:10]:
    print(f"  {a:>10.2f} {b:>10.2f} {c:>10.2f} {auc:>10.4f}")

print(f"\n  Best grid: a={best_grid_weights[0]:.2f}, "
      f"b={best_grid_weights[1]:.2f}, c={best_grid_weights[2]:.2f}, "
      f"AUC={best_grid_auc:.4f}")

# --- Scipy Refinement ---
print(f"\n  Scipy Refinement (Nelder-Mead):")


def neg_blend_auc(params):
    """Negative AUC for minimization. params=[a, b], c=1-a-b."""
    a, b = params
    c = 1.0 - a - b
    if a < 0 or b < 0 or c < 0:
        return 1.0  # penalty
    p_blend = a * voting_oof + b * stacking_oof + c * rank_oof
    return -roc_auc_score(y, p_blend)


# Start from grid best
x0 = [best_grid_weights[0], best_grid_weights[1]]
result = minimize(neg_blend_auc, x0, method="Nelder-Mead",
                  options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 1000})

opt_a, opt_b = result.x
opt_c = 1.0 - opt_a - opt_b

# Clamp to [0, 1]
opt_a = np.clip(opt_a, 0, 1)
opt_b = np.clip(opt_b, 0, 1)
opt_c = np.clip(opt_c, 0, 1)
total = opt_a + opt_b + opt_c
opt_a, opt_b, opt_c = opt_a / total, opt_b / total, opt_c / total

opt_auc = blend_auc([opt_a, opt_b, opt_c],
                     voting_oof, stacking_oof, rank_oof, y)

print(f"  Optimized: a={opt_a:.4f}, b={opt_b:.4f}, c={opt_c:.4f}")
print(f"  Optimized AUC: {opt_auc:.4f}")

# Compare with individual methods
print(f"\n  Comparison:")
print(f"    Voting only:   {voting_auc:.4f}")
print(f"    Stacking only: {stacking_auc:.4f}")
print(f"    Rank only:     {rank_auc:.4f}")
print(f"    Linear blend:  {opt_auc:.4f} ({opt_auc - voting_auc:+.4f} vs Voting)")

# Final linear blend weights
BLEND_WEIGHTS = (opt_a, opt_b, opt_c)
linear_blend_oof = opt_a * voting_oof + opt_b * stacking_oof + opt_c * rank_oof


# ============================================================
# Phase 4: Gated/Selective Blend
# ============================================================
print("\n" + "=" * 60)
print("Phase 4: Gated Blend (Disagreement-Based Routing)")
print("=" * 60)
print("  High disagreement -> Rank-heavy (rescue Stacking FN)")
print("  Low disagreement  -> Linear blend")

best_gated_auc = -1
best_gated_params = None

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ALPHAS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

disagreement = np.abs(stacking_oof - rank_oof)

print(f"\n  Disagreement stats: mean={disagreement.mean():.3f}, "
      f"std={disagreement.std():.3f}, "
      f"min={disagreement.min():.3f}, max={disagreement.max():.3f}")

print(f"\n  {'Threshold':>10} {'Alpha':>8} {'N_disagree':>12} {'AUC':>10} {'vs Linear':>10}")
print("  " + "-" * 54)

for threshold in THRESHOLDS:
    for alpha in ALPHAS:
        disagree_mask = disagreement > threshold
        n_disagree = disagree_mask.sum()

        gated_oof = np.where(
            disagree_mask,
            # Disagreement region: lean toward Rank (rescue Stacking FN)
            alpha * rank_oof + (1 - alpha) * voting_oof,
            # Agreement region: use optimized linear blend
            linear_blend_oof,
        )
        gated_auc = roc_auc_score(y, gated_oof)

        if gated_auc > best_gated_auc:
            best_gated_auc = gated_auc
            best_gated_params = (threshold, alpha)

# Show best gated result
best_th, best_alpha = best_gated_params
disagree_mask = disagreement > best_th
n_disagree = disagree_mask.sum()

print(f"  (showing best only)")
print(f"  {best_th:>10.2f} {best_alpha:>8.2f} {n_disagree:>12d} "
      f"{best_gated_auc:>10.4f} {best_gated_auc - opt_auc:>+10.4f}")

print(f"\n  Best gated: threshold={best_th:.2f}, alpha={best_alpha:.2f}")
print(f"  Gated AUC: {best_gated_auc:.4f} ({best_gated_auc - opt_auc:+.4f} vs Linear)")

# Also show nearby configurations for robustness check
print(f"\n  Top configurations around best:")
top_gated = []
for threshold in THRESHOLDS:
    for alpha in ALPHAS:
        dm = disagreement > threshold
        gated = np.where(
            dm,
            alpha * rank_oof + (1 - alpha) * voting_oof,
            linear_blend_oof,
        )
        top_gated.append((threshold, alpha, roc_auc_score(y, gated)))

top_gated.sort(key=lambda x: -x[2])
for th, al, auc in top_gated[:5]:
    dm = disagreement > th
    print(f"    th={th:.2f}, alpha={al:.1f}, n_disagree={dm.sum()}, "
          f"AUC={auc:.4f} ({auc - opt_auc:+.4f})")


# ============================================================
# Phase 5: Repeated CV Stability Validation
# ============================================================
print("\n" + "=" * 60)
print("Phase 5: Repeated CV Stability (5-fold x 5-seed)")
print("=" * 60)

STABILITY_SEEDS = [42, 123, 456, 789, 2024]

# For each seed, regenerate OOFs and compute blend AUC
seed_blend_aucs = []
seed_voting_aucs = []
seed_weights = []

for s_idx, s in enumerate(STABILITY_SEEDS):
    # Regenerate base OOFs with this seed
    s_base_oof = {}
    for name, fn in MODEL_FNS.items():
        _, _, _, oof_proba, _ = cross_validate_oof(
            fn, X_raw, y, n_folds=N_FOLDS, seed=s, feature_builder=fb,
        )
        s_base_oof[name] = oof_proba

    s_oof_arr = np.column_stack([s_base_oof[n] for n in MODEL_FNS])

    # Voting OOF
    s_voting = s_oof_arr.mean(axis=1)
    s_voting_auc = roc_auc_score(y, s_voting)
    seed_voting_aucs.append(s_voting_auc)

    # Rank OOF
    s_rank = np.column_stack([
        rankdata(s_base_oof[n]) / len(y) for n in MODEL_FNS
    ]).mean(axis=1)

    # Stacking OOF (single seed meta-CV for speed)
    s_stack = np.zeros(len(y))
    s_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=s + 100)
    for _, (tr_idx, va_idx) in enumerate(s_cv.split(s_oof_arr, y)):
        meta = LogisticRegression(C=best_meta_C, random_state=SEED, max_iter=2000)
        meta.fit(s_oof_arr[tr_idx], y.values[tr_idx])
        s_stack[va_idx] = meta.predict_proba(s_oof_arr[va_idx])[:, 1]

    # Apply fixed blend weights (from Phase 3)
    s_blend = opt_a * s_voting + opt_b * s_stack + opt_c * s_rank
    s_blend_auc = roc_auc_score(y, s_blend)
    seed_blend_aucs.append(s_blend_auc)

    # Also re-optimize weights for this seed
    def neg_auc_s(params):
        a, b = params
        c = 1.0 - a - b
        if a < 0 or b < 0 or c < 0:
            return 1.0
        return -roc_auc_score(y, a * s_voting + b * s_stack + c * s_rank)

    res = minimize(neg_auc_s, [opt_a, opt_b], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-6})
    sa, sb = np.clip(res.x, 0, 1)
    sc = max(0, 1.0 - sa - sb)
    t = sa + sb + sc
    seed_weights.append((sa / t, sb / t, sc / t))

    print(f"  Seed {s}: Voting={s_voting_auc:.4f}, Blend={s_blend_auc:.4f} "
          f"(weights: a={sa/t:.2f}, b={sb/t:.2f}, c={sc/t:.2f})")

print(f"\n  Voting: mean={np.mean(seed_voting_aucs):.4f} "
      f"+/- {np.std(seed_voting_aucs):.4f}")
print(f"  Blend:  mean={np.mean(seed_blend_aucs):.4f} "
      f"+/- {np.std(seed_blend_aucs):.4f}")
print(f"  Improvement: {np.mean(seed_blend_aucs) - np.mean(seed_voting_aucs):+.4f}")

# Check weight stability
w_arr = np.array(seed_weights)
print(f"\n  Weight Stability:")
print(f"    a (Voting):   mean={w_arr[:, 0].mean():.3f} +/- {w_arr[:, 0].std():.3f}")
print(f"    b (Stacking): mean={w_arr[:, 1].mean():.3f} +/- {w_arr[:, 1].std():.3f}")
print(f"    c (Rank):     mean={w_arr[:, 2].mean():.3f} +/- {w_arr[:, 2].std():.3f}")


# ============================================================
# Phase 6: Generate Test Predictions and Submission Files
# ============================================================
print("\n" + "=" * 60)
print("Phase 6: Test Predictions and Submissions")
print("=" * 60)

# Build features for full train and test
X_tr_full, X_te_full = fb(X_raw, test)
print(f"  Full train features: {X_tr_full.shape}")
print(f"  Full test features:  {X_te_full.shape}")

# Train 4 base models on ALL training data -> predict test
test_base_preds = {}
for name, fn in MODEL_FNS.items():
    model = fn()
    model.fit(X_tr_full, y)
    pred = model.predict_proba(X_te_full)[:, 1]
    test_base_preds[name] = pred
    print(f"  {name}: test mean={pred.mean():.3f}, std={pred.std():.3f}")

test_base_arr = np.column_stack([test_base_preds[n] for n in MODEL_FNS])

# --- Voting test prediction ---
voting_test = test_base_arr.mean(axis=1)

# --- Rank test prediction ---
rank_test = np.column_stack([
    rankdata(test_base_preds[n]) / len(test) for n in MODEL_FNS
]).mean(axis=1)

# --- Stacking test prediction ---
# Train meta-learner on full OOF (base model OOFs on all training data)
meta_final = LogisticRegression(C=best_meta_C, random_state=SEED, max_iter=2000)
meta_final.fit(oof_arr, y)
stacking_test = meta_final.predict_proba(test_base_arr)[:, 1]

print(f"\n  Meta-learner weights (C={best_meta_C}):")
for name, w in zip(MODEL_FNS.keys(), meta_final.coef_[0]):
    print(f"    {name}: {w:.4f}")
print(f"    intercept: {meta_final.intercept_[0]:.4f}")

# --- Linear Blend test prediction ---
a, b, c = BLEND_WEIGHTS
linear_blend_test = a * voting_test + b * stacking_test + c * rank_test

# --- Gated Blend test prediction ---
test_disagreement = np.abs(stacking_test - rank_test)
gated_blend_test = np.where(
    test_disagreement > best_th,
    best_alpha * rank_test + (1 - best_alpha) * voting_test,
    linear_blend_test,
)

print(f"\n  Test Prediction Summary:")
print(f"    Voting:       mean={voting_test.mean():.3f}, std={voting_test.std():.3f}")
print(f"    Stacking:     mean={stacking_test.mean():.3f}, std={stacking_test.std():.3f}")
print(f"    Rank:         mean={rank_test.mean():.3f}, std={rank_test.std():.3f}")
print(f"    Linear Blend: mean={linear_blend_test.mean():.3f}, std={linear_blend_test.std():.3f}")
print(f"    Gated Blend:  mean={gated_blend_test.mean():.3f}, std={gated_blend_test.std():.3f}")

# Test prediction correlations
print(f"\n  Test Prediction Correlations:")
for n1, p1, n2, p2 in [
    ("Voting", voting_test, "Stacking", stacking_test),
    ("Voting", voting_test, "Rank", rank_test),
    ("Stacking", stacking_test, "Rank", rank_test),
    ("Linear", linear_blend_test, "Gated", gated_blend_test),
]:
    print(f"    {n1}-{n2}: {np.corrcoef(p1, p2)[0, 1]:.4f}")

# --- Save Submissions ---
submissions = {
    "submit_voting_baseline.csv": voting_test,
    "submit_stacking.csv": stacking_test,
    "submit_rank_ensemble.csv": rank_test,
    "submit_linear_blend.csv": linear_blend_test,
    "submit_gated_blend.csv": gated_blend_test,
}

print(f"\n  Saving submissions:")
for fname, pred in submissions.items():
    sub = sample_submit.copy()
    sub[1] = pred
    sub.to_csv(fname, header=None)
    print(f"    {fname}")


# ============================================================
# Phase 7: Evaluate Against Local Ground Truth
# ============================================================
print("\n" + "=" * 60)
print("Phase 7: Local Ground Truth Evaluation")
print("=" * 60)

eval_results = {}
for fname in submissions:
    try:
        result = evaluate_submission(fname)
        eval_results[fname] = result["auc"]
        print(f"  {fname}: Local Public AUC = {result['auc']:.4f}")
    except Exception as e:
        print(f"  {fname}: evaluation failed ({e})")

# Compare with existing best
existing_best = 0.8762  # submit_retuned_voting.csv from exp23
print(f"\n  Existing best (exp23 retuned voting): {existing_best:.4f}")
for fname, auc in sorted(eval_results.items(), key=lambda x: -x[1]):
    diff = auc - existing_best
    marker = " *** NEW BEST" if diff > 0 else ""
    print(f"  {fname}: {auc:.4f} ({diff:+.4f}){marker}")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\n  OOF AUC (Cross-Validation):")
print(f"    Voting:       {voting_auc:.4f}")
print(f"    Stacking:     {stacking_auc:.4f}")
print(f"    Rank:         {rank_auc:.4f}")
print(f"    Linear Blend: {opt_auc:.4f} ({opt_auc - voting_auc:+.4f} vs Voting)")
print(f"    Gated Blend:  {best_gated_auc:.4f} ({best_gated_auc - voting_auc:+.4f} vs Voting)")

print(f"\n  Optimal Weights:")
print(f"    Linear: a(Vote)={opt_a:.4f}, b(Stack)={opt_b:.4f}, c(Rank)={opt_c:.4f}")
print(f"    Gated:  threshold={best_th:.2f}, alpha={best_alpha:.2f}")

print(f"\n  Stability (Repeated CV, 5 seeds):")
print(f"    Voting mean AUC:  {np.mean(seed_voting_aucs):.4f} +/- {np.std(seed_voting_aucs):.4f}")
print(f"    Blend mean AUC:   {np.mean(seed_blend_aucs):.4f} +/- {np.std(seed_blend_aucs):.4f}")

if eval_results:
    best_sub = max(eval_results, key=eval_results.get)
    best_local_auc = eval_results[best_sub]
    print(f"\n  Best submission: {best_sub}")
    print(f"  Local Public AUC: {best_local_auc:.4f}")
    if best_local_auc > existing_best:
        print(f"  NEW BEST! (+{best_local_auc - existing_best:.4f} over exp23)")
    else:
        print(f"  vs existing best: {best_local_auc - existing_best:+.4f}")

print("\nDone!")
