import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# v0: original preprocess (preserved)
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing pipeline."""
    df = df.copy()

    # Age: fill missing with median
    if "age" in df.columns:
        df["age"] = df["age"].fillna(df["age"].median())

    # Fare: fill missing with median
    if "fare" in df.columns:
        df["fare"] = df["fare"].fillna(df["fare"].median())

    # Embarked: fill missing with mode
    if "embarked" in df.columns:
        df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    # Sex: encode
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({"male": 0, "female": 1})

    return df


# ---------------------------------------------------------------------------
# v1: leak-safe preprocess with richer features
# ---------------------------------------------------------------------------
def compute_train_stats(df: pd.DataFrame) -> dict:
    """Compute statistics from train data for leak-free imputation.

    Parameters
    ----------
    df : pd.DataFrame
        Training dataframe (before preprocessing).

    Returns
    -------
    dict with keys:
        age_median_by_pclass : dict[int, float]
        fare_median_by_pclass : dict[int, float]
        embarked_mode : str
    """
    stats = {}
    stats["age_median_by_pclass"] = df.groupby("pclass")["age"].median().to_dict()
    stats["fare_median_by_pclass"] = df.groupby("pclass")["fare"].median().to_dict()
    stats["embarked_mode"] = df["embarked"].mode()[0]
    # fallback medians (when pclass is missing from stats)
    stats["age_median_global"] = df["age"].median()
    stats["fare_median_global"] = df["fare"].median()
    return stats


def preprocess_v1(df: pd.DataFrame, train_stats: dict) -> pd.DataFrame:
    """Leak-safe preprocessing using pre-computed train statistics.

    - age/fare: fill with pclass-wise median (MAR-aware)
    - embarked: fill with train mode
    - sex: encode male=0, female=1
    """
    df = df.copy()

    # Age: pclass-wise median imputation
    if "age" in df.columns:
        for pclass, med in train_stats["age_median_by_pclass"].items():
            mask = df["age"].isna() & (df["pclass"] == pclass)
            df.loc[mask, "age"] = med
        # fallback for any remaining NaN
        df["age"] = df["age"].fillna(train_stats["age_median_global"])

    # Fare: pclass-wise median imputation
    if "fare" in df.columns:
        for pclass, med in train_stats["fare_median_by_pclass"].items():
            mask = df["fare"].isna() & (df["pclass"] == pclass)
            df.loc[mask, "fare"] = med
        df["fare"] = df["fare"].fillna(train_stats["fare_median_global"])

    # Embarked: mode imputation
    if "embarked" in df.columns:
        df["embarked"] = df["embarked"].fillna(train_stats["embarked_mode"])

    # Sex: encode
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({"male": 0, "female": 1})

    return df


def make_features(df: pd.DataFrame, version: str = "v1") -> pd.DataFrame:
    """Create engineered features.

    v0: embarked one-hot only
    v1: family_size, is_alone, log_fare, fare_per_person, pclass_sex + embarked one-hot
    """
    df = df.copy()

    # embarked one-hot (common to v0 and v1)
    if "embarked" in df.columns:
        dummies = pd.get_dummies(df["embarked"], prefix="embarked").astype(int)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=["embarked"])

    if version == "v0":
        return df

    # --- v1 features ---
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    df["log_fare"] = np.log1p(df["fare"])
    df["fare_per_person"] = df["fare"] / df["family_size"]
    df["pclass_sex"] = df["pclass"] * 10 + df["sex"]  # e.g. 10=1class+female, 31=3class+male

    return df


def get_feature_columns(version: str = "v1") -> list[str]:
    """Return the list of feature column names for the given version."""
    if version == "v0":
        return [
            "pclass", "sex", "age", "sibsp", "parch", "fare",
            "embarked_C", "embarked_Q", "embarked_S",
        ]
    # v1
    return [
        "pclass", "sex", "age", "sibsp", "parch", "fare",
        "embarked_C", "embarked_Q", "embarked_S",
        "family_size", "is_alone", "log_fare", "fare_per_person", "pclass_sex",
    ]


def build_pipeline(
    df: pd.DataFrame,
    version: str = "v1",
    train_stats: dict | None = None,
) -> pd.DataFrame:
    """End-to-end: preprocess -> make_features -> select columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe (train or test).
    version : str
        Feature version ("v0" or "v1").
    train_stats : dict or None
        Output of compute_train_stats(). Required for v1.
        If None and version=="v0", uses legacy preprocess().
    """
    # Preprocess
    if version == "v1" and train_stats is not None:
        df = preprocess_v1(df, train_stats)
    else:
        df = preprocess(df)

    # Feature engineering
    df = make_features(df, version=version)

    # Select columns (fill missing one-hot cols with 0)
    cols = get_feature_columns(version)
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    df = df[cols]

    return df


def make_feature_builder(version: str = "v1", extra_fn=None):
    """Generate a leak-free feature_builder for cross-validation.

    Parameters
    ----------
    version : str
        Feature version passed to build_pipeline.
    extra_fn : callable or None
        (df: pd.DataFrame) -> pd.DataFrame.
        Appends extra features to the build_pipeline output.

    Returns
    -------
    builder : callable
        (X_train_raw, X_val_raw) -> (X_train, X_val)
    """
    def builder(X_train_raw, X_val_raw):
        train_stats = compute_train_stats(X_train_raw)
        X_train = build_pipeline(X_train_raw, version=version, train_stats=train_stats)
        X_val = build_pipeline(X_val_raw, version=version, train_stats=train_stats)
        if extra_fn is not None:
            X_train = extra_fn(X_train)
            X_val = extra_fn(X_val)
            for c in X_train.columns:
                if c not in X_val.columns:
                    X_val[c] = 0
            X_val = X_val[X_train.columns]
        return X_train, X_val
    return builder
