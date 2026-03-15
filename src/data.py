import pandas as pd
from src.config import TRAIN_CSV, TEST_CSV, SAMPLE_SUBMIT_CSV, TEST_GROUND_TRUTH_CSV


def load_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV)


def load_test() -> pd.DataFrame:
    return pd.read_csv(TEST_CSV)


def load_sample_submit() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_SUBMIT_CSV)


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_train(), load_test(), load_sample_submit()


def load_test_ground_truth() -> pd.DataFrame:
    return pd.read_csv(TEST_GROUND_TRUTH_CSV)


def load_submission(path: str) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=["id", "prob"])
