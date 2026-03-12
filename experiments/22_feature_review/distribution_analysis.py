"""Train vs Test Distribution Analysis + Covariate Shift Correction.

Phase 1: train/test の特徴量分布を比較
Phase 2: 分布の差が大きい特徴量を特定
Phase 3: Importance Weighting で補正して学習
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
from sklearn.model_selection import cross_val_predict
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy import stats

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

# ============================================================
# Phase 1: Distribution Comparison
# ============================================================
print("=" * 60)
print("Phase 1: Train vs Test Feature Distribution Comparison")
print("=" * 60)

print(f"\n{'Feature':<20} {'Train Mean':>10} {'Test Mean':>10} {'Diff':>8} {'KS stat':>8} {'KS p':>8}")
print("-" * 70)

ks_results = {}
for col in X_train.columns:
    tr_vals = X_train[col].values
    te_vals = X_test[col].values
    tr_mean = tr_vals.mean()
    te_mean = te_vals.mean()
    diff = te_mean - tr_mean

    # Kolmogorov-Smirnov test: tests if two distributions are different
    ks_stat, ks_p = stats.ks_2samp(tr_vals, te_vals)
    ks_results[col] = {"stat": ks_stat, "p": ks_p, "diff": diff}

    flag = " ***" if ks_p < 0.05 else ""
    print(f"  {col:<20} {tr_mean:>8.3f} {te_mean:>8.3f} {diff:>+8.3f} {ks_stat:>8.3f} {ks_p:>8.3f}{flag}")

sig_features = [col for col, r in ks_results.items() if r["p"] < 0.05]
print(f"\n  Significant distribution differences (p<0.05): {sig_features}")
if not sig_features:
    print("  → train/test の分布に統計的に有意な差はない！")


# ============================================================
# Phase 2: Categorical Feature Comparison
# ============================================================
print("\n" + "=" * 60)
print("Phase 2: Categorical Feature Distribution")
print("=" * 60)

# Compare raw categorical features
for col in ["pclass", "sex", "embarked_C", "embarked_Q", "embarked_S"]:
    tr_dist = X_train[col].value_counts(normalize=True).sort_index()
    te_dist = X_test[col].value_counts(normalize=True).sort_index()
    print(f"\n  {col}:")
    for val in sorted(set(list(tr_dist.index) + list(te_dist.index))):
        tr_pct = tr_dist.get(val, 0) * 100
        te_pct = te_dist.get(val, 0) * 100
        print(f"    {val}: train={tr_pct:.1f}%  test={te_pct:.1f}%  diff={te_pct-tr_pct:+.1f}%")


# ============================================================
# Phase 3: Density Ratio Estimation (Importance Weighting)
# ============================================================
print("\n" + "=" * 60)
print("Phase 3: Importance Weighting via Density Ratio")
print("=" * 60)

# Train a classifier to distinguish train vs test
# P(test | x) / P(train | x) gives the importance weight
X_combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
y_domain = np.array([0] * len(X_train) + [1] * len(X_test))  # 0=train, 1=test

# Use LightGBM for density ratio estimation
domain_clf = LGBMClassifier(
    n_estimators=100, learning_rate=0.05, num_leaves=8,
    random_state=SEED, verbose=-1,
)

# Get out-of-fold probabilities for the domain classifier
from sklearn.model_selection import cross_val_predict
domain_proba = cross_val_predict(
    domain_clf, X_combined, y_domain,
    cv=5, method="predict_proba",
)[:, 1]

# Domain classifier accuracy
domain_pred = (domain_proba > 0.5).astype(int)
domain_acc = (domain_pred == y_domain).mean()
print(f"  Domain classifier accuracy: {domain_acc:.4f}")
print(f"  (0.50 = train/test are identical, >0.55 = noticeable shift)")

if domain_acc < 0.55:
    print("  → train/test の分布はほぼ同じ。重み補正の効果は限定的だろう。")
else:
    print("  → 分布の差あり！重み補正の余地がある。")

# Compute importance weights for training samples
train_proba = domain_proba[:len(X_train)]
# w(x) = P(test|x) / P(train|x) = p / (1-p)
# Clip to avoid extreme weights
epsilon = 0.01
weights = np.clip(train_proba / (1 - train_proba + epsilon), 0.1, 10.0)

print(f"\n  Weight statistics:")
print(f"    min={weights.min():.3f}, max={weights.max():.3f}, "
      f"mean={weights.mean():.3f}, std={weights.std():.3f}")

# Show which training samples get high/low weights
weight_df = pd.DataFrame({
    "weight": weights,
    "pclass": X_train["pclass"].values,
    "sex": X_train["sex"].values,
    "age": X_train["age"].values,
})
print(f"\n  Average weight by pclass:")
for pc in sorted(weight_df["pclass"].unique()):
    avg_w = weight_df[weight_df["pclass"] == pc]["weight"].mean()
    print(f"    pclass={int(pc)}: avg_weight={avg_w:.3f}")

print(f"\n  Average weight by sex:")
for s in sorted(weight_df["sex"].unique()):
    label = "female" if s == 1 else "male"
    avg_w = weight_df[weight_df["sex"] == s]["weight"].mean()
    print(f"    {label}: avg_weight={avg_w:.3f}")


# ============================================================
# Phase 4: Train with Importance Weights
# ============================================================
print("\n" + "=" * 60)
print("Phase 4: Training with Importance Weights")
print("=" * 60)

# Best params from Optuna
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

# Baseline (no weights)
print("\n  [Baseline - no weights]")
_, base_m = cross_validate(make_tuned_voting, X_train, y)
print(f"    Voting AUC: {base_m['auc']:.4f}")

# Weighted training - models that support sample_weight
# XGB and LGBM support sample_weight natively
# For weighted CV, we need a custom approach
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Test 1: Weighted LGBM
print("\n  [Weighted LGBM]")
aucs = []
for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y)):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    w_tr = weights[tr_idx]

    model = make_tuned_lgbm()
    model.fit(X_tr, y_tr, sample_weight=w_tr)
    proba = model.predict_proba(X_va)[:, 1]
    aucs.append(roc_auc_score(y_va, proba))
print(f"    LGBM AUC: {np.mean(aucs):.4f} (baseline: {cross_validate(make_tuned_lgbm, X_train, y)[1]['auc']:.4f})")

# Test 2: Weighted XGB
print("\n  [Weighted XGB]")
aucs = []
for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y)):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    w_tr = weights[tr_idx]

    model = make_tuned_xgb()
    model.fit(X_tr, y_tr, sample_weight=w_tr)
    proba = model.predict_proba(X_va)[:, 1]
    aucs.append(roc_auc_score(y_va, proba))
print(f"    XGB AUC: {np.mean(aucs):.4f} (baseline: {cross_validate(make_tuned_xgb, X_train, y)[1]['auc']:.4f})")

# Test 3: Weighted Voting (manual)
print("\n  [Weighted Voting (all models with sample_weight)]")
aucs = []
for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y)):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    w_tr = weights[tr_idx]

    # Train each model with weights where supported
    models = []

    # LogReg with weights
    lr = make_tuned_logreg()
    lr.fit(X_tr, y_tr, model__sample_weight=w_tr)
    models.append(lr)

    # RF with weights
    rf = make_tuned_rf()
    rf.fit(X_tr, y_tr, sample_weight=w_tr)
    models.append(rf)

    # XGB with weights
    xgb = make_tuned_xgb()
    xgb.fit(X_tr, y_tr, sample_weight=w_tr)
    models.append(xgb)

    # LGBM with weights
    lgbm = make_tuned_lgbm()
    lgbm.fit(X_tr, y_tr, sample_weight=w_tr)
    models.append(lgbm)

    # Soft voting (average probabilities)
    probas = np.mean([m.predict_proba(X_va)[:, 1] for m in models], axis=0)
    aucs.append(roc_auc_score(y_va, probas))

weighted_voting_auc = np.mean(aucs)
print(f"    Voting AUC: {weighted_voting_auc:.4f} (baseline: {base_m['auc']:.4f})")
print(f"    Diff: {weighted_voting_auc - base_m['auc']:+.4f}")


# ============================================================
# Phase 5: Combined - Best Features + Importance Weights
# ============================================================
print("\n" + "=" * 60)
print("Phase 5: Best Features (domain+missing) + Importance Weights")
print("=" * 60)

from src.exp_features import make_exp_builder

# domain+missing config
fb = make_exp_builder(
    missing_flags=True, age_bins=None, fare_bins=None,
    interactions=False, polynomial=False, group_stats=False,
    freq_encoding=False, rank_features=False, domain_features=True,
)

# First, baseline with domain+missing features (no weights)
_, dm_base_m = cross_validate(make_tuned_voting, X_raw, y, feature_builder=fb)
print(f"  domain+missing baseline: Voting AUC={dm_base_m['auc']:.4f}")

# Then with importance weights
# Need to recompute weights for the expanded feature set
X_dm_train, X_dm_test = fb(X_raw, test)

# Recompute domain weights with expanded features
X_dm_combined = pd.concat([X_dm_train, X_dm_test], axis=0).reset_index(drop=True)
domain_clf2 = LGBMClassifier(
    n_estimators=100, learning_rate=0.05, num_leaves=8,
    random_state=SEED, verbose=-1,
)
domain_proba2 = cross_val_predict(
    domain_clf2, X_dm_combined, y_domain,
    cv=5, method="predict_proba",
)[:, 1]
weights2 = np.clip(
    domain_proba2[:len(X_dm_train)] / (1 - domain_proba2[:len(X_dm_train)] + epsilon),
    0.1, 10.0,
)

print(f"  Domain classifier accuracy (expanded): "
      f"{((domain_proba2 > 0.5).astype(int) == y_domain).mean():.4f}")

# Weighted Voting with domain+missing features
aucs = []
for fold, (tr_idx, va_idx) in enumerate(cv.split(X_raw, y)):
    X_tr_raw, X_va_raw = X_raw.iloc[tr_idx], X_raw.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    X_tr, X_va = fb(X_tr_raw, X_va_raw)
    w_tr = weights2[tr_idx]

    models = []
    lr = make_tuned_logreg()
    lr.fit(X_tr, y_tr, model__sample_weight=w_tr)
    models.append(lr)

    rf = make_tuned_rf()
    rf.fit(X_tr, y_tr, sample_weight=w_tr)
    models.append(rf)

    xgb = make_tuned_xgb()
    xgb.fit(X_tr, y_tr, sample_weight=w_tr)
    models.append(xgb)

    lgbm = make_tuned_lgbm()
    lgbm.fit(X_tr, y_tr, sample_weight=w_tr)
    models.append(lgbm)

    probas = np.mean([m.predict_proba(X_va)[:, 1] for m in models], axis=0)
    aucs.append(roc_auc_score(y_va, probas))

combo_auc = np.mean(aucs)
print(f"  domain+missing + weights: Voting AUC={combo_auc:.4f}")
print(f"  vs domain+missing baseline: {combo_auc - dm_base_m['auc']:+.4f}")
print(f"  vs v1 baseline: {combo_auc - base_m['auc']:+.4f}")


# ============================================================
# Phase 6: Generate Submission
# ============================================================
print("\n" + "=" * 60)
print("Phase 6: Submission Generation")
print("=" * 60)

sample_submit = pd.read_csv(SAMPLE_SUBMIT_CSV, index_col=0, header=None)

# If weighted voting improved, generate submission
if weighted_voting_auc > base_m["auc"]:
    # Weighted v1 voting
    models = []
    lr = make_tuned_logreg()
    lr.fit(X_train, y, model__sample_weight=weights)
    models.append(lr)

    rf = make_tuned_rf()
    rf.fit(X_train, y, sample_weight=weights)
    models.append(rf)

    xgb_m = make_tuned_xgb()
    xgb_m.fit(X_train, y, sample_weight=weights)
    models.append(xgb_m)

    lgbm_m = make_tuned_lgbm()
    lgbm_m.fit(X_train, y, sample_weight=weights)
    models.append(lgbm_m)

    pred = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)
    sub = sample_submit.copy()
    sub[1] = pred
    sub.to_csv("submit_weighted_v1_voting.csv", header=None)
    print(f"  submit_weighted_v1_voting.csv: mean={pred.mean():.3f}")

# Best combo: domain+missing features + importance weights
X_dm_train_full, X_dm_test_full = fb(X_raw, test)
models = []
lr = make_tuned_logreg()
lr.fit(X_dm_train_full, y, model__sample_weight=weights2)
models.append(lr)

rf = make_tuned_rf()
rf.fit(X_dm_train_full, y, sample_weight=weights2)
models.append(rf)

xgb_m = make_tuned_xgb()
xgb_m.fit(X_dm_train_full, y, sample_weight=weights2)
models.append(xgb_m)

lgbm_m = make_tuned_lgbm()
lgbm_m.fit(X_dm_train_full, y, sample_weight=weights2)
models.append(lgbm_m)

pred = np.mean([m.predict_proba(X_dm_test_full)[:, 1] for m in models], axis=0)
sub = sample_submit.copy()
sub[1] = pred
sub.to_csv("submit_domain_missing_weighted_voting.csv", header=None)
print(f"  submit_domain_missing_weighted_voting.csv: mean={pred.mean():.3f}")

print("\nDone!")
