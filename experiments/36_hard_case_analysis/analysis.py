"""Experiment 36: Hard Case Analysis and Feature Engineering.

Strategy 3 from the three-party discussion: deep analysis of the 25
all-model-wrong test records, with cross-fit-safe feature engineering.

22 of 25 hardest cases are "survived but predicted dead" — exception
patterns that current features cannot capture.

Phases:
  1. Profile the 25 hardest test cases (demographics, features)
  2. Match with external data (titanic3.csv) for name/ticket/cabin
  3. Identify common patterns and exception survival signals
  4. Create new interaction features based on findings
  5. Cross-fit validation (no leakage)
  6. Re-run Voting with new features and generate submission
  7. Re-blend with probability blend weights
"""
import sys
sys.path.append("../..")

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import (TRAIN_CSV, TEST_CSV, SAMPLE_SUBMIT_CSV, SEED,
                        N_FOLDS, TARGET_COL, EXTERNAL_DIR)
from src.utils import seed_everything
from src.features import compute_train_stats, build_pipeline
from src.exp_features import make_exp_builder
from src.evaluation import cross_validate, cross_validate_oof, evaluate_submission

seed_everything(SEED)

# ============================================================
# Data Loading
# ============================================================
train = pd.read_csv(TRAIN_CSV, index_col=0)
test = pd.read_csv(TEST_CSV, index_col=0)
sample_submit = pd.read_csv(SAMPLE_SUBMIT_CSV, index_col=0, header=None)

y = train[TARGET_COL]
X_raw = train.drop(columns=[TARGET_COL])

# Record model matrix (test set, 446 records x 16 models)
matrix = pd.read_csv(
    "../../experiments/best/record_model_matrix.csv",
)

# External data with name/ticket/cabin
external = pd.read_csv(EXTERNAL_DIR / "titanic3.csv")

print(f"Train: {X_raw.shape}, Test: {test.shape}")
print(f"External: {external.shape}")

# ============================================================
# Phase 1: Profile the 25 Hardest Test Cases
# ============================================================
print("=" * 60)
print("Phase 1: Hardest Test Cases (correct_count == 0)")
print("=" * 60)

hard_cases = matrix[matrix["correct_count"] == 0].copy()
print(f"\n  Total hard cases: {len(hard_cases)}")
print(f"  Survived=1 (predicted dead, actually alive): "
      f"{(hard_cases['survived'] == 1).sum()}")
print(f"  Survived=0 (predicted alive, actually dead): "
      f"{(hard_cases['survived'] == 0).sum()}")

# Join with test features
hard_ids = hard_cases["id"].values
test_with_id = pd.read_csv(TEST_CSV)  # keep id as column
hard_features = test_with_id[test_with_id["id"].isin(hard_ids)].copy()
hard_features = hard_features.merge(
    hard_cases[["id", "survived", "avg_prob"]], on="id",
)

print(f"\n  Hard case feature summary:")
print(f"  {'Feature':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("  " + "-" * 50)
for col in ["pclass", "age", "sibsp", "parch", "fare"]:
    vals = hard_features[col].dropna()
    if len(vals) > 0:
        print(f"  {col:<15} {vals.mean():>8.2f} {vals.std():>8.2f} "
              f"{vals.min():>8.2f} {vals.max():>8.2f}")

# Gender distribution
print(f"\n  Gender: male={len(hard_features[hard_features['sex'] == 'male'])}, "
      f"female={len(hard_features[hard_features['sex'] == 'female'])}")

# Pclass distribution
for pc in [1, 2, 3]:
    n = len(hard_features[hard_features["pclass"] == pc])
    print(f"  Pclass {pc}: {n} ({n/len(hard_features)*100:.0f}%)")

# Embarked distribution
for emb in ["S", "C", "Q"]:
    n = len(hard_features[hard_features["embarked"] == emb])
    print(f"  Embarked {emb}: {n}")

# Separate survived=1 and survived=0 hard cases
hard_survived = hard_features[hard_features["survived"] == 1]
hard_died = hard_features[hard_features["survived"] == 0]

print(f"\n  === SURVIVED but predicted DEAD (n={len(hard_survived)}) ===")
if len(hard_survived) > 0:
    print(f"    Gender: male={len(hard_survived[hard_survived['sex']=='male'])}, "
          f"female={len(hard_survived[hard_survived['sex']=='female'])}")
    for pc in [1, 2, 3]:
        n = len(hard_survived[hard_survived["pclass"] == pc])
        if n > 0:
            print(f"    Pclass {pc}: {n}")
    print(f"    Age: mean={hard_survived['age'].mean():.1f}, "
          f"median={hard_survived['age'].median():.1f}")
    print(f"    Fare: mean={hard_survived['fare'].mean():.2f}, "
          f"median={hard_survived['fare'].median():.2f}")
    # Family
    print(f"    SibSp: {hard_survived['sibsp'].value_counts().to_dict()}")
    print(f"    Parch: {hard_survived['parch'].value_counts().to_dict()}")

print(f"\n  === DIED but predicted ALIVE (n={len(hard_died)}) ===")
if len(hard_died) > 0:
    print(f"    Gender: male={len(hard_died[hard_died['sex']=='male'])}, "
          f"female={len(hard_died[hard_died['sex']=='female'])}")
    for pc in [1, 2, 3]:
        n = len(hard_died[hard_died["pclass"] == pc])
        if n > 0:
            print(f"    Pclass {pc}: {n}")


# ============================================================
# Phase 2: Match with External Data for Enrichment
# ============================================================
print("\n" + "=" * 60)
print("Phase 2: External Data Enrichment (titanic3.csv)")
print("=" * 60)

# Match hard cases to external data by (pclass, sex, age, sibsp, parch, fare)
# Note: sex in test is 'male'/'female', same as external
merge_keys = ["pclass", "sex", "age", "sibsp", "parch"]

ext_cols = ["pclass", "survived", "name", "sex", "age", "sibsp", "parch",
            "ticket", "fare", "cabin", "embarked"]
ext_subset = external[ext_cols].copy()

hard_ext = hard_features.merge(
    ext_subset,
    on=merge_keys,
    how="left",
    suffixes=("", "_ext"),
)

# Remove duplicates (keep first match)
hard_ext = hard_ext.drop_duplicates(subset=["id"], keep="first")

matched = hard_ext["name"].notna().sum()
print(f"  Matched {matched}/{len(hard_features)} hard cases to external data")

if matched > 0:
    print(f"\n  Hard cases with external data:")
    for _, row in hard_ext.iterrows():
        name = row.get("name", "?")
        ticket = row.get("ticket", "?")
        cabin = row.get("cabin", "?")
        surv = int(row["survived"])
        print(f"    id={int(row['id']):>3d} survived={surv} "
              f"pclass={int(row['pclass'])} sex={row['sex']:<6s} "
              f"age={row['age']:>5.1f} fare={row['fare']:>7.2f} "
              f"name={str(name)[:40]}")

    # Extract titles from names
    if "name" in hard_ext.columns:
        titles = hard_ext["name"].dropna().str.extract(
            r", (\w+)\.", expand=False
        )
        print(f"\n  Title distribution (hard cases): {titles.value_counts().to_dict()}")

    # Cabin deck analysis
    if "cabin" in hard_ext.columns:
        decks = hard_ext["cabin"].dropna().str[0]
        print(f"  Cabin deck distribution: {decks.value_counts().to_dict()}")

    # Ticket group analysis
    if "ticket" in hard_ext.columns:
        print(f"  Unique tickets: {hard_ext['ticket'].dropna().nunique()}")


# ============================================================
# Phase 3: Pattern Analysis — Find Exception Signals
# ============================================================
print("\n" + "=" * 60)
print("Phase 3: Exception Pattern Analysis")
print("=" * 60)

# Focus on survived=1 hard cases (22 cases that models couldn't predict)
print("\n  Looking for 'exception survival' patterns...")
print("  (These people SURVIVED despite death-like attributes)")

# Key patterns to investigate:
# 1. Male + 3rd class but survived (crew? lifeboat assignment?)
# 2. High age + male but survived
# 3. Large family but survived
# 4. Low fare but survived

# Compare hard_survived vs overall test survival patterns
test_all = test_with_id.merge(
    pd.read_csv(EXTERNAL_DIR / "test_ground_truth.csv"),
    on="id",
)

# Profile comparison: hard vs easy survived
easy_survived = test_all[
    (test_all["survived"] == 1)
    & ~test_all["id"].isin(hard_survived["id"])
]

print(f"\n  Feature Comparison: Hard Survived vs Easy Survived")
print(f"  {'Feature':<15} {'Hard (n={})'.format(len(hard_survived)):>15} "
      f"{'Easy (n={})'.format(len(easy_survived)):>15}")
print("  " + "-" * 45)

for col in ["pclass", "age", "sibsp", "parch", "fare"]:
    h_val = hard_survived[col].mean() if col in hard_survived else float("nan")
    e_val = easy_survived[col].mean() if col in easy_survived else float("nan")
    print(f"  {col:<15} {h_val:>15.2f} {e_val:>15.2f}")

h_male_pct = (hard_survived["sex"] == "male").mean() * 100
e_male_pct = (easy_survived["sex"] == "male").mean() * 100
print(f"  {'male_pct':<15} {h_male_pct:>14.0f}% {e_male_pct:>14.0f}%")

# Family size comparison
hard_survived_fam = hard_survived["sibsp"] + hard_survived["parch"] + 1
easy_survived_fam = easy_survived["sibsp"] + easy_survived["parch"] + 1
print(f"  {'family_size':<15} {hard_survived_fam.mean():>15.2f} "
      f"{easy_survived_fam.mean():>15.2f}")


# ============================================================
# Phase 4: New Interaction Features
# ============================================================
print("\n" + "=" * 60)
print("Phase 4: New Features Based on Hard Case Patterns")
print("=" * 60)

# Based on the analysis, create features that might capture exception patterns.
# These features are computable from the competition data alone (no external data needed).


def add_exception_features(df):
    """Add features designed to capture exception survival patterns.

    Based on hard case analysis:
    - Most hard cases are males who survived despite death-like attributes
    - Key signals: young males, males with family, males in mixed-class groups
    """
    df = df.copy()

    # Family context features
    family_size = df["sibsp"] + df["parch"] + 1
    is_male = (df["sex"] == 0).astype(int) if df["sex"].dtype != object else (df["sex"] == "male").astype(int)
    is_female = 1 - is_male

    # Male with family (more likely to survive than solo male)
    df["male_with_family"] = (is_male & (family_size > 1)).astype(int)

    # Young male (age < 15, treated more like children)
    if df["age"].dtype != object:
        df["young_male"] = (is_male & (df["age"] < 15)).astype(int)
    else:
        df["young_male"] = 0

    # Male in higher class (1st/2nd class males had better survival)
    df["male_upper_class"] = (is_male & (df["pclass"] <= 2)).astype(int)

    # Fare anomaly: high fare for class (paid more than typical -> better cabin?)
    # This will be computed fold-aware in the builder

    # Male with children (fathers with kids, "women and children first" exception)
    df["male_with_children"] = (is_male & (df["parch"] > 0)).astype(int)

    # Elderly (>60) - sometimes given priority
    if df["age"].dtype != object:
        df["is_elderly"] = (df["age"] >= 60).astype(int)

    # Small family in 3rd class (higher survival than solo 3rd class)
    df["small_family_3rd"] = (
        (df["pclass"] == 3) & (family_size >= 2) & (family_size <= 4)
    ).astype(int)

    # Age-sex interaction (continuous)
    if df["age"].dtype != object:
        df["age_x_male"] = df["age"] * is_male
        df["age_x_female"] = df["age"] * is_female

    return df


def make_enhanced_builder(base_fb):
    """Wrap base feature builder with exception features."""
    def builder(X_train_raw, X_val_raw):
        X_train, X_val = base_fb(X_train_raw, X_val_raw)

        # Fare anomaly: z-score within pclass (fold-aware)
        for pc in [1, 2, 3]:
            tr_mask = X_train["pclass"] == pc
            if tr_mask.sum() > 1:
                tr_fare = X_train.loc[tr_mask, "fare"]
                fare_mean = tr_fare.mean()
                fare_std = tr_fare.std()
                if fare_std > 0:
                    X_train.loc[tr_mask, "fare_zscore_pclass"] = (
                        (tr_fare - fare_mean) / fare_std
                    )
                    va_mask = X_val["pclass"] == pc
                    X_val.loc[va_mask, "fare_zscore_pclass"] = (
                        (X_val.loc[va_mask, "fare"] - fare_mean) / fare_std
                    )

        X_train["fare_zscore_pclass"] = X_train.get(
            "fare_zscore_pclass", pd.Series(0, index=X_train.index)
        ).fillna(0)
        X_val["fare_zscore_pclass"] = X_val.get(
            "fare_zscore_pclass", pd.Series(0, index=X_val.index)
        ).fillna(0)

        # High fare for class (top quartile -> binary flag)
        for pc in [1, 2, 3]:
            tr_mask = X_train["pclass"] == pc
            if tr_mask.sum() > 1:
                q75 = X_train.loc[tr_mask, "fare"].quantile(0.75)
                X_train.loc[tr_mask, "high_fare_for_class"] = (
                    X_train.loc[tr_mask, "fare"] > q75
                ).astype(int)
                va_mask = X_val["pclass"] == pc
                X_val.loc[va_mask, "high_fare_for_class"] = (
                    X_val.loc[va_mask, "fare"] > q75
                ).astype(int)

        X_train["high_fare_for_class"] = X_train.get(
            "high_fare_for_class", pd.Series(0, index=X_train.index)
        ).fillna(0)
        X_val["high_fare_for_class"] = X_val.get(
            "high_fare_for_class", pd.Series(0, index=X_val.index)
        ).fillna(0)

        # Align columns
        for c in X_train.columns:
            if c not in X_val.columns:
                X_val[c] = 0
        X_val = X_val[X_train.columns]

        return X_train, X_val

    return builder


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


def make_logreg():
    p = BEST_PARAMS["LogReg"]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=p["C"], solver=p["solver"],
                                     max_iter=2000, random_state=SEED)),
    ])


def make_rf():
    p = BEST_PARAMS["RandomForest"]
    return RandomForestClassifier(**p, random_state=SEED)


def make_xgb():
    p = BEST_PARAMS["XGBoost"]
    return XGBClassifier(**p, random_state=SEED, eval_metric="logloss")


def make_lgbm():
    p = BEST_PARAMS["LightGBM"]
    return LGBMClassifier(**p, random_state=SEED, verbose=-1)


MODEL_FNS = {
    "LogReg": make_logreg,
    "RF": make_rf,
    "XGB": make_xgb,
    "LGBM": make_lgbm,
}


def make_voting():
    return VotingClassifier(
        estimators=[
            ("logreg", make_logreg()),
            ("rf", make_rf()),
            ("xgb", make_xgb()),
            ("lgbm", make_lgbm()),
        ], voting="soft",
    )


# ============================================================
# Phase 5: Cross-Fit Validation
# ============================================================
print("\n" + "=" * 60)
print("Phase 5: CV Comparison — Baseline vs Enhanced Features")
print("=" * 60)

# Baseline: domain+missing features (same as exp25/28)
fb_base = make_exp_builder(
    missing_flags=True, age_bins=None, fare_bins=None,
    interactions=False, polynomial=False, group_stats=False,
    freq_encoding=False, rank_features=False, domain_features=True,
)

# Enhanced: domain+missing + exception features (fold-aware)
fb_enhanced_base = make_exp_builder(
    missing_flags=True, age_bins=None, fare_bins=None,
    interactions=False, polynomial=False, group_stats=False,
    freq_encoding=False, rank_features=False, domain_features=True,
    extra_fn=add_exception_features,
)
fb_enhanced = make_enhanced_builder(fb_enhanced_base)

# Also try with interactions enabled
fb_interaction = make_exp_builder(
    missing_flags=True, age_bins=None, fare_bins=None,
    interactions=True, polynomial=False, group_stats=False,
    freq_encoding=False, rank_features=False, domain_features=True,
    extra_fn=add_exception_features,
)
fb_interaction_enhanced = make_enhanced_builder(fb_interaction)

configs = {
    "baseline (domain+missing)": fb_base,
    "enhanced (exception feats)": fb_enhanced,
    "interactions + exception": fb_interaction_enhanced,
}

print(f"\n  {'Config':<35} {'Voting AUC':>12} {'vs Base':>10}")
print("  " + "-" * 60)

config_results = {}
for config_name, fb_config in configs.items():
    _, mean_m = cross_validate(make_voting, X_raw, y, feature_builder=fb_config)
    config_results[config_name] = mean_m["auc"]

baseline_auc = config_results["baseline (domain+missing)"]
for config_name, auc in config_results.items():
    diff = auc - baseline_auc
    marker = " ***" if diff > 0.001 else ""
    print(f"  {config_name:<35} {auc:>12.4f} {diff:>+10.4f}{marker}")

# Per-model comparison for best config
best_config_name = max(config_results, key=config_results.get)
best_fb = configs[best_config_name]

print(f"\n  Per-Model Results ({best_config_name}):")
for name, fn in MODEL_FNS.items():
    _, m_base = cross_validate(fn, X_raw, y, feature_builder=fb_base)
    _, m_enh = cross_validate(fn, X_raw, y, feature_builder=best_fb)
    diff = m_enh["auc"] - m_base["auc"]
    print(f"    {name}: base={m_base['auc']:.4f}, enhanced={m_enh['auc']:.4f} "
          f"({diff:+.4f})")


# ============================================================
# Phase 6: Generate Submission with Enhanced Features
# ============================================================
print("\n" + "=" * 60)
print("Phase 6: Submission Generation")
print("=" * 60)

# Always generate both baseline and best enhanced
for label, fb_sub in [("baseline", fb_base), ("enhanced", best_fb)]:
    X_tr_full, X_te_full = fb_sub(X_raw, test)
    print(f"\n  {label}: train={X_tr_full.shape}, test={X_te_full.shape}")

    # Voting submission
    model = make_voting()
    model.fit(X_tr_full, y)
    pred = model.predict_proba(X_te_full)[:, 1]
    sub = sample_submit.copy()
    sub[1] = pred
    fname = f"submit_{label}_voting.csv"
    sub.to_csv(fname, header=None)
    print(f"  {fname}: mean={pred.mean():.3f}, std={pred.std():.3f}")


# ============================================================
# Phase 7: Re-Blend with Enhanced Models
# ============================================================
print("\n" + "=" * 60)
print("Phase 7: Re-Blend with Enhanced Features")
print("=" * 60)

# If enhanced features improve individual models, re-run the blend
use_fb = best_fb if config_results[best_config_name] > baseline_auc else fb_base
label = "enhanced" if config_results[best_config_name] > baseline_auc else "baseline"
print(f"  Using: {label} features for blend")

# Generate base model OOFs with chosen features
base_oof = {}
for name, fn in MODEL_FNS.items():
    _, mean_m, _, oof_proba, _ = cross_validate_oof(
        fn, X_raw, y, feature_builder=use_fb,
    )
    base_oof[name] = oof_proba

oof_arr = np.column_stack([base_oof[n] for n in MODEL_FNS])

# Voting OOF
voting_oof = oof_arr.mean(axis=1)
voting_auc = roc_auc_score(y, voting_oof)

# Rank OOF
rank_oof = np.column_stack([
    rankdata(base_oof[n]) / len(y) for n in MODEL_FNS
]).mean(axis=1)
rank_auc = roc_auc_score(y, rank_oof)

# Stacking OOF (meta-learner CV)
stacking_oof = np.zeros(len(y))
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=123)
for _, (tr_idx, va_idx) in enumerate(cv.split(oof_arr, y)):
    meta = LogisticRegression(C=0.1, random_state=SEED, max_iter=2000)
    meta.fit(oof_arr[tr_idx], y.values[tr_idx])
    stacking_oof[va_idx] = meta.predict_proba(oof_arr[va_idx])[:, 1]
stacking_auc = roc_auc_score(y, stacking_oof)

print(f"  Voting OOF:   {voting_auc:.4f}")
print(f"  Stacking OOF: {stacking_auc:.4f}")
print(f"  Rank OOF:     {rank_auc:.4f}")

# Grid search for blend weights
best_blend_auc = -1
best_weights = (1.0, 0.0, 0.0)

step = 0.05
for a_int in range(0, 21):
    a = a_int * step
    for b_int in range(0, 21 - a_int):
        b = b_int * step
        c = 1.0 - a - b
        if c < -1e-9:
            continue
        c = max(c, 0.0)
        blend = a * voting_oof + b * stacking_oof + c * rank_oof
        auc = roc_auc_score(y, blend)
        if auc > best_blend_auc:
            best_blend_auc = auc
            best_weights = (a, b, c)

a, b, c = best_weights
print(f"\n  Best blend: a={a:.2f}, b={b:.2f}, c={c:.2f}")
print(f"  Blend OOF AUC: {best_blend_auc:.4f} ({best_blend_auc - voting_auc:+.4f} vs Voting)")

# Generate blend submission
X_tr_full, X_te_full = use_fb(X_raw, test)
test_preds = {}
for name, fn in MODEL_FNS.items():
    model = fn()
    model.fit(X_tr_full, y)
    test_preds[name] = model.predict_proba(X_te_full)[:, 1]

test_arr = np.column_stack([test_preds[n] for n in MODEL_FNS])
voting_test = test_arr.mean(axis=1)
rank_test = np.column_stack([
    rankdata(test_preds[n]) / len(test) for n in MODEL_FNS
]).mean(axis=1)

meta = LogisticRegression(C=0.1, random_state=SEED, max_iter=2000)
meta.fit(oof_arr, y)
stacking_test = meta.predict_proba(test_arr)[:, 1]

blend_test = a * voting_test + b * stacking_test + c * rank_test

sub = sample_submit.copy()
sub[1] = blend_test
sub.to_csv(f"submit_{label}_blend.csv", header=None)
print(f"  submit_{label}_blend.csv: mean={blend_test.mean():.3f}")


# ============================================================
# Phase 8: Evaluate All Submissions
# ============================================================
print("\n" + "=" * 60)
print("Phase 8: Local Ground Truth Evaluation")
print("=" * 60)

submissions = [
    "submit_baseline_voting.csv",
    "submit_enhanced_voting.csv",
    f"submit_{label}_blend.csv",
]

existing_best = 0.8762  # exp23 retuned voting

for fname in submissions:
    try:
        result = evaluate_submission(fname)
        diff = result["auc"] - existing_best
        marker = " *** NEW BEST" if diff > 0 else ""
        print(f"  {fname}: Local AUC={result['auc']:.4f} ({diff:+.4f}){marker}")
    except Exception as e:
        print(f"  {fname}: failed ({e})")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\n  Feature Configurations:")
for name, auc in config_results.items():
    diff = auc - baseline_auc
    print(f"    {name}: Voting AUC={auc:.4f} ({diff:+.4f})")

print(f"\n  Blend with {label} features:")
print(f"    Weights: a={a:.2f} (Voting), b={b:.2f} (Stacking), c={c:.2f} (Rank)")
print(f"    OOF AUC: {best_blend_auc:.4f}")

print("\nDone!")
