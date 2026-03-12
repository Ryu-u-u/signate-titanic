"""Feature Review: Permutation Importance + Experimental Feature Search.

Phase 1: Permutation Importance で v1 の 14 特徴量を評価
Phase 2: 実験的特徴量をチューニング済みモデルで再テスト
Phase 3: 特徴量の取捨選択（弱い特徴量の除去 + 強い実験的特徴量の追加）
Phase 4: ベスト特徴量セットで提出ファイル生成
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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import TRAIN_CSV, TEST_CSV, SAMPLE_SUBMIT_CSV, SEED, N_FOLDS, TARGET_COL
from src.utils import seed_everything
from src.features import compute_train_stats, build_pipeline, get_feature_columns
from src.exp_features import make_exp_builder, EXP_PRESETS
from src.evaluation import cross_validate, get_cv_splitter

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

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Features: {list(X_train.columns)}")

# ============================================================
# Best hyperparameters from Optuna (experiments/21_tuning)
# ============================================================
# Exact values from Optuna tuning (experiments/21_tuning)
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


# ============================================================
# Phase 1: Permutation Importance
# ============================================================
print("\n" + "=" * 60)
print("Phase 1: Permutation Importance Analysis")
print("=" * 60)

# Use CV-based permutation importance for robustness
cv = get_cv_splitter()
importances = {feat: [] for feat in X_train.columns}

for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y)):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # Use RF as importance estimator (most stable)
    model = make_tuned_rf()
    model.fit(X_tr, y_tr)
    perm = permutation_importance(model, X_va, y_va,
                                  scoring="roc_auc", n_repeats=10,
                                  random_state=SEED)
    for i, feat in enumerate(X_train.columns):
        importances[feat].append(perm.importances_mean[i])

print(f"\n{'Feature':<20} {'Mean PI':>10} {'Std PI':>10} {'Verdict':>10}")
print("-" * 55)
imp_df = pd.DataFrame({
    "feature": list(importances.keys()),
    "mean": [np.mean(v) for v in importances.values()],
    "std": [np.std(v) for v in importances.values()],
})
imp_df = imp_df.sort_values("mean", ascending=False).reset_index(drop=True)
for _, row in imp_df.iterrows():
    verdict = "STRONG" if row["mean"] > 0.01 else ("weak" if row["mean"] > 0.001 else "DROP?")
    print(f"  {row['feature']:<20} {row['mean']:>8.4f} {row['std']:>10.4f}   {verdict}")


# ============================================================
# Phase 2: Baseline v1 with tuned models (reference)
# ============================================================
print("\n" + "=" * 60)
print("Phase 2: Baseline v1 (14 features) - Reference")
print("=" * 60)

baseline_results = {}
for name, fn in [
    ("RF", make_tuned_rf),
    ("XGB", make_tuned_xgb),
    ("LGBM", make_tuned_lgbm),
    ("Voting", make_tuned_voting),
]:
    _, mean_m = cross_validate(fn, X_train, y)
    baseline_results[name] = mean_m["auc"]
    print(f"  {name}: AUC={mean_m['auc']:.4f}")


# ============================================================
# Phase 3: Feature Ablation (weak feature removal)
# ============================================================
print("\n" + "=" * 60)
print("Phase 3: Feature Ablation (remove weak features)")
print("=" * 60)

# Identify features to potentially drop (PI < 0.001)
weak_features = imp_df[imp_df["mean"] < 0.001]["feature"].tolist()
print(f"  Weak features (PI < 0.001): {weak_features}")

if weak_features:
    X_train_reduced = X_train.drop(columns=weak_features)
    X_test_reduced = X_test.drop(columns=weak_features)
    print(f"  Reduced: {X_train_reduced.shape[1]} features (dropped {len(weak_features)})")

    for name, fn in [("RF", make_tuned_rf), ("Voting", make_tuned_voting)]:
        _, mean_m = cross_validate(fn, X_train_reduced, y)
        diff = mean_m["auc"] - baseline_results[name]
        print(f"  {name}: AUC={mean_m['auc']:.4f} (diff={diff:+.4f})")
else:
    print("  No weak features found. Skipping ablation.")
    X_train_reduced = X_train
    X_test_reduced = X_test


# ============================================================
# Phase 4: Experimental Features with Tuned Models
# ============================================================
print("\n" + "=" * 60)
print("Phase 4: Experimental Features (with tuned models)")
print("=" * 60)

# Test each category individually + presets
exp_configs = {
    "missing_flags_only": dict(
        missing_flags=True, age_bins=None, fare_bins=None,
        interactions=False, polynomial=False, group_stats=False,
        freq_encoding=False, rank_features=False, domain_features=False,
    ),
    "domain_only": dict(
        missing_flags=False, age_bins=None, fare_bins=None,
        interactions=False, polynomial=False, group_stats=False,
        freq_encoding=False, rank_features=False, domain_features=True,
    ),
    "interactions_only": dict(
        missing_flags=False, age_bins=None, fare_bins=None,
        interactions=True, polynomial=False, group_stats=False,
        freq_encoding=False, rank_features=False, domain_features=False,
    ),
    "bins_only": dict(
        missing_flags=False, age_bins="rule", fare_bins="quantile",
        interactions=False, polynomial=False, group_stats=False,
        freq_encoding=False, rank_features=False, domain_features=False,
    ),
    "group_stats_only": dict(
        missing_flags=False, age_bins=None, fare_bins=None,
        interactions=False, polynomial=False, group_stats=True,
        freq_encoding=False, rank_features=False, domain_features=False,
    ),
    "domain+missing": dict(
        missing_flags=True, age_bins=None, fare_bins=None,
        interactions=False, polynomial=False, group_stats=False,
        freq_encoding=False, rank_features=False, domain_features=True,
    ),
    "domain+interactions": dict(
        missing_flags=False, age_bins=None, fare_bins=None,
        interactions=True, polynomial=False, group_stats=False,
        freq_encoding=False, rank_features=False, domain_features=True,
    ),
    "recommended": EXP_PRESETS["recommended"],
    "kitchen_sink": EXP_PRESETS["kitchen_sink"],
}

exp_results = {}
print(f"\n{'Config':<25} {'RF AUC':>10} {'XGB AUC':>10} {'Vote AUC':>10}")
print("-" * 60)

for config_name, kwargs in exp_configs.items():
    fb = make_exp_builder(**kwargs)

    results = {}
    for model_name, fn in [
        ("RF", make_tuned_rf),
        ("XGB", make_tuned_xgb),
        ("Voting", make_tuned_voting),
    ]:
        _, mean_m = cross_validate(fn, X_raw, y, feature_builder=fb)
        results[model_name] = mean_m["auc"]

    exp_results[config_name] = results
    print(f"  {config_name:<25} {results['RF']:>8.4f} {results['XGB']:>8.4f} {results['Voting']:>8.4f}")

# Reference baseline
print(f"  {'[baseline v1]':<25} {baseline_results['RF']:>8.4f} {baseline_results['XGB']:>8.4f} {baseline_results['Voting']:>8.4f}")


# ============================================================
# Phase 5: Best config deep-dive
# ============================================================
print("\n" + "=" * 60)
print("Phase 5: Best Configurations Deep-Dive")
print("=" * 60)

# Find configs that beat baseline for Voting
baseline_voting = baseline_results["Voting"]
improvements = []
for config_name, results in exp_results.items():
    diff = results["Voting"] - baseline_voting
    improvements.append((config_name, results["Voting"], diff))

improvements.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'Config':<25} {'Voting AUC':>12} {'vs Baseline':>12}")
print("-" * 55)
for name, auc, diff in improvements:
    marker = " ***" if diff > 0 else ""
    print(f"  {name:<25} {auc:>10.4f} {diff:>+10.4f}{marker}")
print(f"  {'[baseline v1]':<25} {baseline_voting:>10.4f}       ---")

# If there's a winner, do a thorough evaluation
best_config_name = improvements[0][0]
best_config_auc = improvements[0][1]
best_config_diff = improvements[0][2]

if best_config_diff > 0:
    print(f"\n  Best config: {best_config_name} (Voting AUC={best_config_auc:.4f}, +{best_config_diff:.4f})")

    # Full evaluation with all models
    fb = make_exp_builder(**exp_configs[best_config_name])
    print("\n  Full model evaluation with best config:")
    for model_name, fn in [
        ("LogReg", make_tuned_logreg),
        ("RF", make_tuned_rf),
        ("XGB", make_tuned_xgb),
        ("LGBM", make_tuned_lgbm),
        ("Voting", make_tuned_voting),
    ]:
        _, mean_m = cross_validate(fn, X_raw, y, feature_builder=fb)
        print(f"    {model_name}: AUC={mean_m['auc']:.4f}, Acc={mean_m['accuracy']:.4f}")
else:
    print(f"\n  No improvement found. Baseline v1 remains best.")


# ============================================================
# Phase 6: Feature Selection via forward/backward
# ============================================================
print("\n" + "=" * 60)
print("Phase 6: Selective Feature Addition")
print("=" * 60)

# Try adding individual experimental features to v1 baseline
# Focus on domain features since they're interpretable and less likely to overfit
from src.exp_features import (
    _add_domain_features, _add_age_rule_bins, _capture_missing,
    _add_missing_flags, _add_interactions
)

individual_features = {
    "is_child": lambda df: df.assign(is_child=(df["age"] <= 12).astype(int)),
    "is_mother": lambda df: df.assign(
        is_mother=((df["sex"] == 1) & (df["parch"] > 0) & (df["age"] > 18)).astype(int)
    ),
    "fare_zero": lambda df: df.assign(fare_zero=(df["fare"] == 0).astype(int)),
    "family_small": lambda df: df.assign(
        family_small=df["family_size"].between(2, 4).astype(int)
    ),
    "family_large": lambda df: df.assign(
        family_large=(df["family_size"] >= 5).astype(int)
    ),
    "age_bin": lambda df: _add_age_rule_bins(df.copy()),
    "age_pclass": lambda df: df.assign(age_pclass=df["age"] * df["pclass"]),
    "fare_pclass": lambda df: df.assign(fare_pclass=df["fare"] * df["pclass"]),
}

print(f"\n{'Added Feature':<20} {'Voting AUC':>12} {'vs Baseline':>12}")
print("-" * 50)

feature_gains = []
for feat_name, transform_fn in individual_features.items():
    X_tr_aug = transform_fn(X_train.copy())
    X_te_aug = transform_fn(X_test.copy())
    _, mean_m = cross_validate(make_tuned_voting, X_tr_aug, y)
    diff = mean_m["auc"] - baseline_voting
    feature_gains.append((feat_name, mean_m["auc"], diff))
    marker = " ***" if diff > 0 else ""
    print(f"  {feat_name:<20} {mean_m['auc']:>10.4f} {diff:>+10.4f}{marker}")

print(f"  {'[baseline v1]':<20} {baseline_voting:>10.4f}       ---")

# Combine beneficial features
beneficial = [name for name, _, diff in feature_gains if diff > 0.001]
print(f"\n  Beneficial features (diff > +0.001): {beneficial}")

if beneficial:
    def add_beneficial(df):
        for name in beneficial:
            df = individual_features[name](df)
        return df

    X_tr_best = add_beneficial(X_train.copy())
    X_te_best = add_beneficial(X_test.copy())

    print(f"  Combined ({len(beneficial)} features added, total={X_tr_best.shape[1]}):")
    for model_name, fn in [
        ("RF", make_tuned_rf),
        ("XGB", make_tuned_xgb),
        ("LGBM", make_tuned_lgbm),
        ("Voting", make_tuned_voting),
    ]:
        _, mean_m = cross_validate(fn, X_tr_best, y)
        diff = mean_m["auc"] - baseline_results.get(model_name, 0)
        print(f"    {model_name}: AUC={mean_m['auc']:.4f} (diff={diff:+.4f})")


# ============================================================
# Phase 7: Generate Submissions (if improved)
# ============================================================
print("\n" + "=" * 60)
print("Phase 7: Submission Generation")
print("=" * 60)

sample_submit = pd.read_csv(SAMPLE_SUBMIT_CSV, index_col=0, header=None)

# Always generate: v1 baseline (reference)
model = make_tuned_voting()
model.fit(X_train, y)
pred = model.predict_proba(X_test)[:, 1]
sub = sample_submit.copy()
sub[1] = pred
sub.to_csv("submit_v1_voting.csv", header=None)
print(f"  submit_v1_voting.csv (baseline): mean={pred.mean():.3f}")

# Generate with best improvement if found
# Check if selective addition was better
selective_best = max(feature_gains, key=lambda x: x[1]) if feature_gains else None
if selective_best and selective_best[2] > 0:
    print(f"\n  Best single feature addition: {selective_best[0]} (AUC={selective_best[1]:.4f})")

if beneficial:
    X_tr_best = add_beneficial(X_train.copy())
    X_te_best = add_beneficial(X_test.copy())
    model = make_tuned_voting()
    model.fit(X_tr_best, y)
    pred = model.predict_proba(X_te_best)[:, 1]
    sub = sample_submit.copy()
    sub[1] = pred
    sub.to_csv("submit_v1plus_voting.csv", header=None)
    print(f"  submit_v1plus_voting.csv (v1+{beneficial}): mean={pred.mean():.3f}")

# If exp config was better
if best_config_diff > 0:
    fb = make_exp_builder(**exp_configs[best_config_name])
    # Need to retrain with full data using the builder
    full_stats = compute_train_stats(X_raw)
    X_full = build_pipeline(X_raw, version="v1", train_stats=full_stats)
    X_te_full = build_pipeline(test, version="v1", train_stats=full_stats)

    # Apply exp features (using full train stats)
    # For submission, we use full train data for stats computation
    fb_submit = make_exp_builder(**exp_configs[best_config_name])
    X_tr_exp, X_te_exp = fb_submit(X_raw, test)

    model = make_tuned_voting()
    model.fit(X_tr_exp, y)
    pred = model.predict_proba(X_te_exp)[:, 1]
    sub = sample_submit.copy()
    sub[1] = pred
    safe_name = best_config_name.replace("+", "_")
    sub.to_csv(f"submit_{safe_name}_voting.csv", header=None)
    print(f"  submit_{safe_name}_voting.csv: mean={pred.mean():.3f}")

print("\nDone!")
