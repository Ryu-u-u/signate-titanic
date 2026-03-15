from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

TRAIN_CSV = RAW_DIR / "train.csv"
TEST_CSV = RAW_DIR / "test.csv"
SAMPLE_SUBMIT_CSV = RAW_DIR / "sample_submit.csv"

EXTERNAL_DIR = DATA_DIR / "external"
TEST_GROUND_TRUTH_CSV = EXTERNAL_DIR / "test_ground_truth.csv"

# === Reproducibility ===
SEED = 42

# === Cross-validation ===
N_FOLDS = 5

# === Target ===
TARGET_COL = "survived"
ID_COL = "id"
