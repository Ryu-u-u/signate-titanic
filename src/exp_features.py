"""Experimental feature catalog for Titanic feature engineering lab.

Usage
-----
>>> from src.exp_features import make_exp_builder, EXP_PRESETS
>>> fb = make_exp_builder(**EXP_PRESETS["recommended"])
>>> results = run_all_models(models, X_raw, y, feature_builder=fb)

Categories
----------
1. Missing flags       — pre-imputation binary indicators
2. Binning             — age rule bins, age/fare quantile bins
3. Interactions        — age*pclass, fare*pclass, sex*pclass dummies, etc.
4. Polynomial          — age^2, fare^2, age*fare
5. Group statistics    — diff-from-median, z-score (fold-aware)
6. Frequency encoding  — pclass frequency (fold-aware)
7. Fare rank           — fare percentile within pclass (fold-aware)
8. Domain features     — is_child, is_mother, fare_zero, family bins
"""

import numpy as np
import pandas as pd

from src.features import compute_train_stats, build_pipeline


# ===================================================================
# 1. Missing Flags  (pre-imputation capture)
# ===================================================================

def _capture_missing(df_raw, cols=("age", "fare", "embarked")):
    """Capture missing masks from raw data before imputation."""
    return {c: df_raw[c].isna().astype(int).values
            for c in cols if c in df_raw.columns}


def _add_missing_flags(df, masks):
    """Add binary missing-indicator columns."""
    for col, vals in masks.items():
        df[f"{col}_missing"] = vals
    return df


# ===================================================================
# 2. Binning
# ===================================================================

_AGE_RULE_EDGES = [0, 12, 18, 35, 60, 100]


def _add_age_rule_bins(df):
    """Domain-knowledge age bins: child(0-12)/teen/adult/middle/senior."""
    df["age_bin"] = pd.cut(
        df["age"], bins=_AGE_RULE_EDGES, labels=False, include_lowest=True,
    )
    df["age_bin"] = df["age_bin"].fillna(2).astype(int)  # fallback: adult
    return df


def _qcut_edges(series, q=5):
    """Compute quantile bin edges from a pandas Series."""
    _, edges = pd.qcut(series.dropna(), q=q, retbins=True, duplicates="drop")
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _apply_bins(df, col, edges, name):
    """Apply pre-computed bin edges to a column."""
    df[name] = pd.cut(df[col], bins=edges, labels=False, include_lowest=True)
    df[name] = df[name].fillna(-1).astype(int)
    return df


# ===================================================================
# 3. Interaction Features
# ===================================================================

def _add_interactions(df):
    """Add interaction features between key variables."""
    df["age_pclass"] = df["age"] * df["pclass"]
    df["fare_pclass"] = df["fare"] * df["pclass"]
    if "log_fare" in df.columns:
        df["log_fare_per_pclass"] = df["log_fare"] / df["pclass"]
    if "family_size" in df.columns:
        df["family_sex"] = df["family_size"] * df["sex"]
    if "is_alone" in df.columns:
        df["alone_sex"] = df["is_alone"] * df["sex"]
    # Sex x pclass one-hot dummies
    for pc in [1, 2, 3]:
        df[f"female_{pc}"] = ((df["sex"] == 1) & (df["pclass"] == pc)).astype(int)
        df[f"male_{pc}"] = ((df["sex"] == 0) & (df["pclass"] == pc)).astype(int)
    return df


# ===================================================================
# 4. Polynomial / Non-linear
# ===================================================================

def _add_polynomial(df):
    """Add polynomial and cross terms."""
    df["age_sq"] = df["age"] ** 2
    df["fare_sq"] = df["fare"] ** 2
    df["age_fare"] = df["age"] * df["fare"]
    return df


# ===================================================================
# 5. Group Statistics  (fold-aware: train-only computation)
# ===================================================================

_GROUP_SPECS = [
    ["pclass"],
    ["sex"],
    ["pclass", "sex"],
]


def _compute_group_stats(X_train, cols=("fare", "age")):
    """Compute group-level statistics from training data only."""
    stats = {}
    for col in cols:
        for groups in _GROUP_SPECS:
            key = f"{col}_{'_'.join(groups)}"
            g = X_train.groupby(groups)[col]
            stats[f"{key}_median"] = g.median().to_dict()
            stats[f"{key}_mean"] = g.mean().to_dict()
            stats[f"{key}_std"] = g.std().fillna(0).to_dict()
        stats[f"{col}_global_mean"] = float(X_train[col].mean())
    return stats


def _group_lookup(df, groups, mapping):
    """Map group-level statistic to individual rows."""
    if len(groups) == 1:
        return df[groups[0]].map(mapping)
    keys = list(zip(*(df[g] for g in groups)))
    return pd.Series([mapping.get(k, np.nan) for k in keys], index=df.index)


def _add_group_stats(df, gstats, cols=("fare", "age")):
    """Apply group statistics as diff-from-median and z-score features."""
    for col in cols:
        fallback = gstats[f"{col}_global_mean"]
        for groups in _GROUP_SPECS:
            key = f"{col}_{'_'.join(groups)}"
            sfx = "_".join(groups)
            med = _group_lookup(df, groups, gstats[f"{key}_median"]).fillna(fallback)
            mn = _group_lookup(df, groups, gstats[f"{key}_mean"]).fillna(fallback)
            sd = _group_lookup(df, groups, gstats[f"{key}_std"]).fillna(1).replace(0, 1)
            df[f"{col}_diff_{sfx}"] = df[col] - med
            df[f"{col}_z_{sfx}"] = (df[col] - mn) / sd
    return df


# ===================================================================
# 6. Frequency Encoding  (fold-aware)
# ===================================================================

def _compute_freq(X_train, cols=("pclass",)):
    """Compute frequency encoding maps from training data."""
    return {c: X_train[c].value_counts(normalize=True).to_dict()
            for c in cols if c in X_train.columns}


def _add_freq(df, freq_maps):
    """Apply frequency encoding."""
    for col, fmap in freq_maps.items():
        df[f"{col}_freq"] = df[col].map(fmap).fillna(0)
    return df


# ===================================================================
# 7. Fare Rank / Percentile  (fold-aware)
# ===================================================================

def _compute_fare_rank_stats(X_train):
    """Compute fare min/max per pclass from training data."""
    stats = {}
    for pc in X_train["pclass"].unique():
        m = X_train["pclass"] == pc
        stats[pc] = (float(X_train.loc[m, "fare"].min()),
                     float(X_train.loc[m, "fare"].max()))
    return stats


def _add_fare_rank(df, rstats):
    """Add fare percentile within pclass (0-1 scaled)."""
    pctile = pd.Series(0.5, index=df.index, dtype=float)
    for pc in df["pclass"].unique():
        m = df["pclass"] == pc
        lo, hi = rstats.get(pc, (0, 1))
        rng = hi - lo if hi != lo else 1.0
        pctile.loc[m] = (df.loc[m, "fare"] - lo) / rng
    df["fare_pctile_in_pclass"] = pctile.clip(0, 1)
    return df


# ===================================================================
# 8. Domain-Knowledge Features
# ===================================================================

def _add_domain_features(df):
    """Titanic-specific domain features."""
    df["is_child"] = (df["age"] <= 12).astype(int)
    df["is_mother"] = (
        (df["sex"] == 1) & (df["parch"] > 0) & (df["age"] > 18)
    ).astype(int)
    df["fare_zero"] = (df["fare"] == 0).astype(int)
    if "family_size" in df.columns:
        df["family_small"] = df["family_size"].between(2, 4).astype(int)
        df["family_large"] = (df["family_size"] >= 5).astype(int)
        df["child_with_family"] = (
            (df["age"] <= 12) & (df["family_size"] > 1)
        ).astype(int)
    return df


# ===================================================================
# Presets
# ===================================================================

EXP_PRESETS = {
    "minimal": dict(
        missing_flags=True, age_bins=None, fare_bins=None,
        interactions=False, polynomial=False, group_stats=False,
        freq_encoding=False, rank_features=False, domain_features=False,
    ),
    "recommended": dict(
        missing_flags=True, age_bins="rule", fare_bins="quantile",
        interactions=True, polynomial=False, group_stats=True,
        freq_encoding=False, rank_features=False, domain_features=True,
    ),
    "kitchen_sink": dict(
        missing_flags=True, age_bins="rule", fare_bins="quantile",
        interactions=True, polynomial=True, group_stats=True,
        freq_encoding=True, rank_features=True, domain_features=True,
    ),
}


# ===================================================================
# Builder Factory
# ===================================================================

def make_exp_builder(
    version="v1",
    missing_flags=True,
    age_bins="rule",
    fare_bins="quantile",
    interactions=True,
    polynomial=False,
    group_stats=True,
    freq_encoding=False,
    rank_features=False,
    domain_features=True,
    n_fare_qbins=5,
    n_age_qbins=5,
    extra_fn=None,
):
    """Create a leak-free feature builder with experimental features.

    All fold-aware statistics (group stats, frequency, rank) are computed
    from the training split only, preventing data leakage in CV.

    Parameters
    ----------
    version : str
        Base feature version ("v0" or "v1").
    missing_flags : bool
        Add age_missing, fare_missing, embarked_missing.
    age_bins : {"rule", "quantile", None}
        "rule": domain-knowledge bins (child/teen/adult/middle/senior).
        "quantile": data-driven quantile bins from train fold.
    fare_bins : {"quantile", None}
        "quantile": data-driven quantile bins from train fold.
    interactions : bool
        Add age*pclass, fare*pclass, sex*pclass dummies, etc.
    polynomial : bool
        Add age^2, fare^2, age*fare.
    group_stats : bool
        Add diff-from-median and z-score within (pclass), (sex),
        (pclass, sex) groups. Computed from train fold only.
    freq_encoding : bool
        Add frequency encoding for pclass.
    rank_features : bool
        Add fare percentile within pclass.
    domain_features : bool
        Add is_child, is_mother, fare_zero, family size bins.
    n_fare_qbins : int
        Number of quantile bins for fare.
    n_age_qbins : int
        Number of quantile bins for age.
    extra_fn : callable or None
        Additional (df) -> df transform applied at the end.

    Returns
    -------
    builder : callable
        (X_train_raw, X_val_raw) -> (X_train, X_val)
    """
    def builder(X_train_raw, X_val_raw):
        # --- Pre-imputation: capture missing masks ---
        tr_miss = _capture_missing(X_train_raw) if missing_flags else {}
        va_miss = _capture_missing(X_val_raw) if missing_flags else {}

        # --- Base pipeline (v0/v1) ---
        stats = compute_train_stats(X_train_raw)
        Xtr = build_pipeline(X_train_raw, version=version, train_stats=stats)
        Xva = build_pipeline(X_val_raw, version=version, train_stats=stats)

        # 1. Missing flags
        if missing_flags:
            Xtr = _add_missing_flags(Xtr, tr_miss)
            Xva = _add_missing_flags(Xva, va_miss)

        # 2. Binning
        if age_bins == "rule":
            Xtr = _add_age_rule_bins(Xtr)
            Xva = _add_age_rule_bins(Xva)
        elif age_bins == "quantile":
            edges = _qcut_edges(Xtr["age"], n_age_qbins)
            Xtr = _apply_bins(Xtr, "age", edges, "age_qbin")
            Xva = _apply_bins(Xva, "age", edges, "age_qbin")

        if fare_bins == "quantile":
            edges = _qcut_edges(Xtr["fare"], n_fare_qbins)
            Xtr = _apply_bins(Xtr, "fare", edges, "fare_qbin")
            Xva = _apply_bins(Xva, "fare", edges, "fare_qbin")

        # 3. Interactions
        if interactions:
            Xtr = _add_interactions(Xtr)
            Xva = _add_interactions(Xva)

        # 4. Polynomial
        if polynomial:
            Xtr = _add_polynomial(Xtr)
            Xva = _add_polynomial(Xva)

        # 5. Group statistics (train-fold only)
        if group_stats:
            gs = _compute_group_stats(Xtr)
            Xtr = _add_group_stats(Xtr, gs)
            Xva = _add_group_stats(Xva, gs)

        # 6. Frequency encoding (train-fold only)
        if freq_encoding:
            fm = _compute_freq(Xtr)
            Xtr = _add_freq(Xtr, fm)
            Xva = _add_freq(Xva, fm)

        # 7. Rank features (train-fold only)
        if rank_features:
            rs = _compute_fare_rank_stats(Xtr)
            Xtr = _add_fare_rank(Xtr, rs)
            Xva = _add_fare_rank(Xva, rs)

        # 8. Domain features
        if domain_features:
            Xtr = _add_domain_features(Xtr)
            Xva = _add_domain_features(Xva)

        # 9. Extra transform
        if extra_fn is not None:
            Xtr = extra_fn(Xtr)
            Xva = extra_fn(Xva)

        # Align columns (handle any missing columns in val)
        for c in Xtr.columns:
            if c not in Xva.columns:
                Xva[c] = 0
        Xva = Xva[Xtr.columns]

        return Xtr, Xva

    return builder
