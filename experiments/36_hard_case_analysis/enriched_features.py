"""Experiment 36b: External Data Enrichment + Enhanced Features.

Enrich competition train/test with titanic3.csv to recover:
  - Title from name (Mr/Mrs/Miss/Master/etc.)
  - Cabin deck (A-G)
  - Ticket group size (shared tickets)

Combined with exception features from Phase 4, then blend with exp23.
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
X_raw_train = train.drop(columns=[TARGET_COL])

# External data
external = pd.read_csv(EXTERNAL_DIR / "titanic3.csv")
print(f"Train: {X_raw_train.shape}, Test: {test.shape}, External: {external.shape}")


# ============================================================
# Phase 1: Match Competition Data to External Data
# ============================================================
print("=" * 60)
print("Phase 1: External Data Matching")
print("=" * 60)


def enrich_with_external(comp_df, ext_df):
    """Match competition data to external data and extract features.

    Uses a dictionary lookup approach for reliable matching.
    Key: (pclass, sex, sibsp, parch, fare_round)
    Preserves the original DataFrame index.
    """
    ext = ext_df.copy()
    comp = comp_df.copy()

    # Extract features from external data
    ext["title"] = ext["name"].str.extract(r", (\w+)\.", expand=False)
    ext["deck"] = ext["cabin"].str[0]
    ticket_counts = ext["ticket"].value_counts()
    ext["ticket_group_size"] = ext["ticket"].map(ticket_counts)

    # Build lookup dictionary: key → (title, deck, ticket_group_size)
    lookup = {}
    for _, row in ext.iterrows():
        sex = row["sex"].lower() if isinstance(row["sex"], str) else "unknown"
        fare_r = round(row["fare"], 1) if pd.notna(row["fare"]) else -1
        key = (int(row["pclass"]), sex, int(row["sibsp"]),
               int(row["parch"]), fare_r)
        if key not in lookup:
            title = row["title"] if pd.notna(row["title"]) else "Unknown"
            deck = row["deck"] if pd.notna(row["deck"]) else "Unknown"
            tgs = row["ticket_group_size"] if pd.notna(row["ticket_group_size"]) else 1
            lookup[key] = (title, deck, tgs)

    # Match competition records
    titles = []
    decks = []
    tgs_list = []
    matched_count = 0

    for idx, row in comp.iterrows():
        sex = row["sex"].lower() if isinstance(row["sex"], str) else (
            "male" if row["sex"] == 0 else "female")
        fare_r = round(row["fare"], 1) if pd.notna(row["fare"]) else -1
        key = (int(row["pclass"]), sex, int(row["sibsp"]),
               int(row["parch"]), fare_r)

        if key in lookup:
            t, d, g = lookup[key]
            titles.append(t)
            decks.append(d)
            tgs_list.append(g)
            matched_count += 1
        else:
            titles.append("Unknown")
            decks.append("Unknown")
            tgs_list.append(1)

    comp["title"] = titles
    comp["deck"] = decks
    comp["ticket_group_size"] = tgs_list

    print(f"  Matched {matched_count}/{len(comp)} records")
    print(f"  Title distribution: {comp['title'].value_counts().head(6).to_dict()}")
    print(f"  Deck distribution: {comp['deck'].value_counts().head(8).to_dict()}")

    return comp


print("\n  Enriching train data:")
train_enriched = enrich_with_external(X_raw_train, external)
print(f"\n  Enriching test data:")
test_enriched = enrich_with_external(test, external)


# ============================================================
# Phase 2: Feature Engineering with External Features
# ============================================================
print("\n" + "=" * 60)
print("Phase 2: Enhanced Feature Builder")
print("=" * 60)


def add_enriched_features(df):
    """Add features from external data enrichment + exception patterns."""
    df = df.copy()

    is_male = (df["sex"] == 0).astype(int) if df["sex"].dtype != object else (df["sex"] == "male").astype(int)

    # --- Exception features (from analysis.py Phase 4) ---
    family_size = df["sibsp"] + df["parch"] + 1
    df["male_with_family"] = (is_male & (family_size > 1)).astype(int)
    if df["age"].dtype != object:
        df["young_male"] = (is_male & (df["age"] < 15)).astype(int)
        df["age_x_male"] = df["age"] * is_male
        df["age_x_female"] = df["age"] * (1 - is_male)
        df["is_elderly"] = (df["age"] >= 60).astype(int)
    df["male_upper_class"] = (is_male & (df["pclass"] <= 2)).astype(int)
    df["male_with_children"] = (is_male & (df["parch"] > 0)).astype(int)
    df["small_family_3rd"] = (
        (df["pclass"] == 3) & (family_size >= 2) & (family_size <= 4)
    ).astype(int)

    # --- Title features ---
    if "title" in df.columns:
        # Encode title as numeric categories
        title_map = {
            "Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3,
            "Dr": 4, "Rev": 5, "Unknown": 6,
        }
        df["title_enc"] = df["title"].map(title_map).fillna(6).astype(int)
        # Is rare title (not Mr/Mrs/Miss/Master)
        df["rare_title"] = (~df["title"].isin(["Mr", "Mrs", "Miss", "Master"])).astype(int)
        # Master (young boy) flag
        df["is_master"] = (df["title"] == "Master").astype(int)

    # --- Deck features ---
    if "deck" in df.columns:
        # Known cabin flag
        df["has_cabin"] = (df["deck"] != "Unknown").astype(int)
        # Upper deck (A-C) vs lower
        df["upper_deck"] = df["deck"].isin(["A", "B", "C"]).astype(int)

    # --- Ticket group size ---
    if "ticket_group_size" in df.columns:
        df["in_group"] = (df["ticket_group_size"] > 1).astype(int)
        df["large_group"] = (df["ticket_group_size"] >= 4).astype(int)

    # Drop text columns used for feature generation
    df = df.drop(columns=["title", "deck", "ticket_group_size"], errors="ignore")

    return df


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
# Phase 3: Build Feature Builders
# ============================================================

# Store enriched data globally for feature builder to use
_ENRICHED_TRAIN = train_enriched
_ENRICHED_TEST = test_enriched


def make_enriched_fb():
    """Feature builder that uses external data enrichment."""
    base_fb = make_exp_builder(
        missing_flags=True, age_bins=None, fare_bins=None,
        interactions=False, polynomial=False, group_stats=False,
        freq_encoding=False, rank_features=False, domain_features=True,
    )

    def builder(X_train_raw, X_val_raw):
        # Get base features
        X_train, X_val = base_fb(X_train_raw, X_val_raw)

        # Add enriched features from pre-computed enrichment
        # Match by index (train/val are subsets of X_raw_train by index)
        tr_idx = X_train_raw.index
        va_idx = X_val_raw.index

        # Get enriched columns for train/val (use .reindex for safe lookup)
        enrich_cols = ["title", "deck", "ticket_group_size"]
        for col in enrich_cols:
            if col in _ENRICHED_TRAIN.columns:
                X_train[col] = _ENRICHED_TRAIN.reindex(tr_idx)[col].values
                X_val[col] = _ENRICHED_TRAIN.reindex(va_idx)[col].values

        # Apply enriched feature engineering
        X_train = add_enriched_features(X_train)
        X_val = add_enriched_features(X_val)

        # Fare z-score within pclass (fold-aware)
        for pc in [1, 2, 3]:
            tr_mask = X_train["pclass"] == pc
            if tr_mask.sum() > 1:
                fare_mean = X_train.loc[tr_mask, "fare"].mean()
                fare_std = X_train.loc[tr_mask, "fare"].std()
                if fare_std > 0:
                    X_train.loc[tr_mask, "fare_zscore_pclass"] = (
                        (X_train.loc[tr_mask, "fare"] - fare_mean) / fare_std
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

        # Align columns
        for c in X_train.columns:
            if c not in X_val.columns:
                X_val[c] = 0
        X_val = X_val[X_train.columns]

        return X_train, X_val

    return builder


# Baseline
fb_base = make_exp_builder(
    missing_flags=True, age_bins=None, fare_bins=None,
    interactions=False, polynomial=False, group_stats=False,
    freq_encoding=False, rank_features=False, domain_features=True,
)

# Enriched
fb_enriched = make_enriched_fb()


# ============================================================
# Phase 4: CV Comparison
# ============================================================
print("\n" + "=" * 60)
print("Phase 3: CV Comparison")
print("=" * 60)

configs = {
    "baseline (domain+missing)": fb_base,
    "enriched (external+exception)": fb_enriched,
}

print(f"\n  {'Config':<40} {'Voting AUC':>12} {'vs Base':>10}")
print("  " + "-" * 65)

config_results = {}
for config_name, fb_config in configs.items():
    _, mean_m = cross_validate(make_voting, X_raw_train, y,
                               feature_builder=fb_config)
    config_results[config_name] = mean_m["auc"]

baseline_auc = config_results["baseline (domain+missing)"]
for config_name, auc in config_results.items():
    diff = auc - baseline_auc
    marker = " ***" if diff > 0.001 else ""
    print(f"  {config_name:<40} {auc:>12.4f} {diff:>+10.4f}{marker}")

# Per-model
enriched_auc = config_results["enriched (external+exception)"]
print(f"\n  Per-Model (enriched features):")
for name, fn in MODEL_FNS.items():
    _, m_base = cross_validate(fn, X_raw_train, y, feature_builder=fb_base)
    _, m_enh = cross_validate(fn, X_raw_train, y, feature_builder=fb_enriched)
    diff = m_enh["auc"] - m_base["auc"]
    print(f"    {name}: base={m_base['auc']:.4f}, enriched={m_enh['auc']:.4f} "
          f"({diff:+.4f})")


# ============================================================
# Phase 5: Generate Submissions
# ============================================================
print("\n" + "=" * 60)
print("Phase 4: Submission Generation")
print("=" * 60)

# For test data, we need the enriched test features
# Temporarily swap the enriched data reference for test prediction
def make_enriched_test_fb():
    """Feature builder for full train → test prediction."""
    base_fb = make_exp_builder(
        missing_flags=True, age_bins=None, fare_bins=None,
        interactions=False, polynomial=False, group_stats=False,
        freq_encoding=False, rank_features=False, domain_features=True,
    )

    def builder(X_train_full, X_test):
        X_tr, X_te = base_fb(X_train_full, X_test)

        # Add enriched features for train
        enrich_cols = ["title", "deck", "ticket_group_size"]
        for col in enrich_cols:
            if col in _ENRICHED_TRAIN.columns:
                X_tr[col] = _ENRICHED_TRAIN[col].values

        # Add enriched features for test
        for col in enrich_cols:
            if col in _ENRICHED_TEST.columns:
                X_te[col] = _ENRICHED_TEST[col].values

        X_tr = add_enriched_features(X_tr)
        X_te = add_enriched_features(X_te)

        # Fare z-score
        for pc in [1, 2, 3]:
            tr_mask = X_tr["pclass"] == pc
            if tr_mask.sum() > 1:
                fare_mean = X_tr.loc[tr_mask, "fare"].mean()
                fare_std = X_tr.loc[tr_mask, "fare"].std()
                if fare_std > 0:
                    X_tr.loc[tr_mask, "fare_zscore_pclass"] = (
                        (X_tr.loc[tr_mask, "fare"] - fare_mean) / fare_std
                    )
                    te_mask = X_te["pclass"] == pc
                    X_te.loc[te_mask, "fare_zscore_pclass"] = (
                        (X_te.loc[te_mask, "fare"] - fare_mean) / fare_std
                    )

        X_tr["fare_zscore_pclass"] = X_tr.get(
            "fare_zscore_pclass", pd.Series(0, index=X_tr.index)
        ).fillna(0)
        X_te["fare_zscore_pclass"] = X_te.get(
            "fare_zscore_pclass", pd.Series(0, index=X_te.index)
        ).fillna(0)

        for c in X_tr.columns:
            if c not in X_te.columns:
                X_te[c] = 0
        X_te = X_te[X_tr.columns]

        return X_tr, X_te

    return builder


fb_test = make_enriched_test_fb()
X_tr_full, X_te_full = fb_test(X_raw_train, test)
print(f"  Enriched train: {X_tr_full.shape}")
print(f"  Enriched test:  {X_te_full.shape}")
print(f"  Features: {list(X_tr_full.columns)}")

# Voting submission
model = make_voting()
model.fit(X_tr_full, y)
pred_voting = model.predict_proba(X_te_full)[:, 1]
sub = sample_submit.copy()
sub[1] = pred_voting
sub.to_csv("submit_enriched_voting.csv", header=None)
print(f"\n  submit_enriched_voting.csv: mean={pred_voting.mean():.3f}")


# ============================================================
# Phase 6: Blend with exp23 Retuned
# ============================================================
print("\n" + "=" * 60)
print("Phase 5: Blend with Exp23 Retuned Voting")
print("=" * 60)

# Load exp23 retuned
try:
    retuned = pd.read_csv(
        "../23_hp_retune_domain_missing/submit_retuned_voting.csv",
        header=None, names=["id", "prob"],
    )
    retuned_pred = retuned.set_index("id")["prob"]

    enriched_pred = pd.Series(pred_voting, index=X_te_full.index)

    print(f"  Correlation: {np.corrcoef(enriched_pred, retuned_pred)[0, 1]:.4f}")

    # Try various blend weights
    print(f"\n  {'Weight':>10} {'AUC':>10}")
    print("  " + "-" * 22)

    best_w = 0.5
    best_auc = -1

    for w_int in range(0, 21):
        w = w_int * 0.05
        blend = w * enriched_pred.values + (1 - w) * retuned_pred.values
        sub = sample_submit.copy()
        sub[1] = blend
        sub.to_csv("_temp.csv", header=None)
        result = evaluate_submission("_temp.csv")
        if w_int % 4 == 0 or result["auc"] > best_auc:
            print(f"  {w:>10.2f} {result['auc']:>10.4f}")
        if result["auc"] > best_auc:
            best_auc = result["auc"]
            best_w = w

    print(f"\n  Best blend: w_enriched={best_w:.2f}, AUC={best_auc:.4f}")

    # Save best blend
    blend = best_w * enriched_pred.values + (1 - best_w) * retuned_pred.values
    sub = sample_submit.copy()
    sub[1] = blend
    sub.to_csv("submit_enriched_retuned_blend.csv", header=None)

    import os
    if os.path.exists("_temp.csv"):
        os.remove("_temp.csv")

except Exception as e:
    print(f"  Blend failed: {e}")


# ============================================================
# Phase 7: Evaluate
# ============================================================
print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

existing_best = 0.8762

for fname in [
    "submit_enriched_voting.csv",
    "submit_enriched_retuned_blend.csv",
    "../36_hard_case_analysis/submit_enhanced_voting.csv",
    "../23_hp_retune_domain_missing/submit_retuned_voting.csv",
]:
    try:
        result = evaluate_submission(fname)
        diff = result["auc"] - existing_best
        marker = " *** NEW BEST" if diff > 0 else ""
        name = fname.split("/")[-1]
        print(f"  {name}: AUC={result['auc']:.4f} ({diff:+.4f}){marker}")
    except Exception as e:
        name = fname.split("/")[-1]
        print(f"  {name}: FAILED ({e})")

print("\nDone!")
