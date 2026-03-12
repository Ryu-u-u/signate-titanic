import pandas as pd


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
