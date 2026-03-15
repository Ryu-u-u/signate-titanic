"""Rank/Copula Ensemble: Percentile-Rank Averaging.

Hypothesis: Converting each model's predictions to percentile ranks before
averaging absorbs scale differences between models, potentially yielding
a better-calibrated ensemble.

Phase 1: Get OOF predictions from each tuned model
Phase 2: Rank-transform and compare with simple averaging
Phase 3: Compare rank ensemble vs Equal Voting
Phase 4: Generate submission
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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import TRAIN_CSV, TEST_CSV, SAMPLE_SUBMIT_CSV, SEED, N_FOLDS, TARGET_COL
from src.utils import seed_everything
from src.features import compute_train_stats, build_pipeline
from src.exp_features import make_exp_builder
from src.evaluation import cross_validate, cross_validate_oof, get_cv_splitter

seed_everything(SEED)

# ============================================================
# Data Loading
# ============================================================
train = pd.read_csv(TRAIN_CSV, index_col=0)
test = pd.read_csv(TEST_CSV, index_col=0)

y = train[TARGET_COL]
X_raw = train.drop(columns=[TARGET_COL])

print(f"X_raw: {X_raw.shape}")

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

# Feature builder: domain+missing
fb = make_exp_builder(
    missing_flags=True, age_bins=None, fare_bins=None,
    interactions=False, polynomial=False, group_stats=False,
    freq_encoding=False, rank_features=False, domain_features=True,
)


# ============================================================
# Model Factories
# ============================================================
def make_tuned_logreg():
    p = BEST_PARAMS["LogReg"]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=p["C"], solver=p["solver"],
            max_iter=2000, random_state=SEED,
        )),
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
# Phase 1: Get OOF Predictions from Each Model
# ============================================================
print("=" * 60)
print("Phase 1: Out-of-Fold Predictions per Model")
print("=" * 60)

model_fns = {
    "LogReg": make_tuned_logreg,
    "RF": make_tuned_rf,
    "XGB": make_tuned_xgb,
    "LGBM": make_tuned_lgbm,
}

oof_probas = {}
oof_models = {}
individual_aucs = {}

for name, fn in model_fns.items():
    _, mean_m, _, oof_proba, models = cross_validate_oof(
        fn, X_raw, y, feature_builder=fb
    )
    oof_probas[name] = oof_proba
    oof_models[name] = models
    individual_aucs[name] = mean_m["auc"]
    print(f"  {name}: AUC={mean_m['auc']:.4f}, Acc={mean_m['accuracy']:.4f}")

# Show prediction distribution per model
print(f"\n  {'Model':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("  " + "-" * 42)
for name, proba in oof_probas.items():
    print(f"  {name:<10} {proba.mean():>8.3f} {proba.std():>8.3f} "
          f"{proba.min():>8.3f} {proba.max():>8.3f}")


# ============================================================
# Phase 2: Rank Transform and Average
# ============================================================
print("\n" + "=" * 60)
print("Phase 2: Rank Transformation and Averaging")
print("=" * 60)

# Rank transform: convert to percentile ranks (0 to 1)
rank_probas = {}
for name, proba in oof_probas.items():
    rank_probas[name] = rankdata(proba) / len(proba)

# Show rank distribution per model
print(f"\n  Rank-transformed distributions:")
print(f"  {'Model':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("  " + "-" * 42)
for name, rproba in rank_probas.items():
    print(f"  {name:<10} {rproba.mean():>8.3f} {rproba.std():>8.3f} "
          f"{rproba.min():>8.3f} {rproba.max():>8.3f}")

# Average raw OOF probabilities (simple average)
avg_simple = np.mean(list(oof_probas.values()), axis=0)
auc_simple = roc_auc_score(y, avg_simple)
print(f"\n  Simple average OOF AUC: {auc_simple:.4f}")

# Average rank-transformed OOF probabilities
avg_rank = np.mean(list(rank_probas.values()), axis=0)
auc_rank = roc_auc_score(y, avg_rank)
print(f"  Rank average OOF AUC:   {auc_rank:.4f}")
print(f"  Difference:             {auc_rank - auc_simple:+.4f}")

# Correlation between raw probabilities across models
print(f"\n  Correlation matrix (raw probabilities):")
prob_df = pd.DataFrame(oof_probas)
corr_matrix = prob_df.corr()
print(corr_matrix.to_string(float_format="%.3f"))

# Correlation between rank-transformed probabilities
print(f"\n  Correlation matrix (rank-transformed):")
rank_df = pd.DataFrame(rank_probas)
rank_corr_matrix = rank_df.corr()
print(rank_corr_matrix.to_string(float_format="%.3f"))


# ============================================================
# Phase 3: Compare Rank Ensemble vs Equal Voting
# ============================================================
print("\n" + "=" * 60)
print("Phase 3: Rank Ensemble vs Equal Voting Comparison")
print("=" * 60)

# Equal Voting baseline (using cross_validate for consistency)
_, voting_m = cross_validate(make_tuned_voting, X_raw, y, feature_builder=fb)
voting_auc = voting_m["auc"]

print(f"\n  {'Method':<30} {'AUC':>10}")
print("  " + "-" * 42)
for name in model_fns:
    print(f"  {name + ' (individual)':<30} {individual_aucs[name]:>10.4f}")
print(f"  {'Simple Average (OOF)':<30} {auc_simple:>10.4f}")
print(f"  {'Rank Average (OOF)':<30} {auc_rank:>10.4f}")
print(f"  {'Equal Voting (CV mean)':<30} {voting_auc:>10.4f}")

# Detailed fold-level comparison using custom CV
print(f"\n  Fold-level comparison (Rank Ensemble vs Voting):")
cv = get_cv_splitter(n_folds=N_FOLDS, seed=SEED)

fold_rank_aucs = []
fold_simple_aucs = []
fold_voting_aucs = []

for fold, (tr_idx, va_idx) in enumerate(cv.split(X_raw, y)):
    X_tr_raw, X_va_raw = X_raw.iloc[tr_idx], X_raw.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    X_tr, X_va = fb(X_tr_raw, X_va_raw)

    # Individual model predictions for this fold
    fold_probas = {}
    for name, fn in model_fns.items():
        model = fn()
        model.fit(X_tr, y_tr)
        fold_probas[name] = model.predict_proba(X_va)[:, 1]

    # Simple average
    fold_simple_avg = np.mean(list(fold_probas.values()), axis=0)
    fold_simple_auc = roc_auc_score(y_va, fold_simple_avg)
    fold_simple_aucs.append(fold_simple_auc)

    # Rank average (rank within validation fold)
    fold_rank_probas = {}
    for name, proba in fold_probas.items():
        fold_rank_probas[name] = rankdata(proba) / len(proba)
    fold_rank_avg = np.mean(list(fold_rank_probas.values()), axis=0)
    fold_rank_auc = roc_auc_score(y_va, fold_rank_avg)
    fold_rank_aucs.append(fold_rank_auc)

    # Voting ensemble
    voting_model = make_tuned_voting()
    voting_model.fit(X_tr, y_tr)
    voting_proba = voting_model.predict_proba(X_va)[:, 1]
    fold_voting_auc = roc_auc_score(y_va, voting_proba)
    fold_voting_aucs.append(fold_voting_auc)

    print(f"    Fold {fold}: Simple={fold_simple_auc:.4f}, "
          f"Rank={fold_rank_auc:.4f}, "
          f"Voting={fold_voting_auc:.4f}")

mean_rank = np.mean(fold_rank_aucs)
mean_simple = np.mean(fold_simple_aucs)
mean_voting = np.mean(fold_voting_aucs)

print(f"\n  Mean AUC:")
print(f"    Simple average:  {mean_simple:.4f} (+/- {np.std(fold_simple_aucs):.4f})")
print(f"    Rank average:    {mean_rank:.4f} (+/- {np.std(fold_rank_aucs):.4f})")
print(f"    Equal Voting:    {mean_voting:.4f} (+/- {np.std(fold_voting_aucs):.4f})")
print(f"    Rank vs Simple:  {mean_rank - mean_simple:+.4f}")
print(f"    Rank vs Voting:  {mean_rank - mean_voting:+.4f}")


# ============================================================
# Phase 4: Generate Submissions
# ============================================================
print("\n" + "=" * 60)
print("Phase 4: Submission Generation")
print("=" * 60)

sample_submit = pd.read_csv(SAMPLE_SUBMIT_CSV, index_col=0, header=None)

# Build features for full train and test
X_tr_full, X_te_full = fb(X_raw, test)
print(f"  Full train features: {X_tr_full.shape}")
print(f"  Full test features:  {X_te_full.shape}")

# Train all 4 models on full data and get test predictions
test_probas = {}
for name, fn in model_fns.items():
    model = fn()
    model.fit(X_tr_full, y)
    pred = model.predict_proba(X_te_full)[:, 1]
    test_probas[name] = pred
    print(f"  {name}: mean={pred.mean():.3f}, std={pred.std():.3f}")

# Rank-transform per model, then average
test_rank_probas = {}
for name, proba in test_probas.items():
    test_rank_probas[name] = rankdata(proba) / len(proba)

avg_rank_test = np.mean(list(test_rank_probas.values()), axis=0)
avg_simple_test = np.mean(list(test_probas.values()), axis=0)

print(f"\n  Simple average: mean={avg_simple_test.mean():.3f}, std={avg_simple_test.std():.3f}")
print(f"  Rank average:   mean={avg_rank_test.mean():.3f}, std={avg_rank_test.std():.3f}")

# Submission: rank ensemble
sub_rank = sample_submit.copy()
sub_rank[1] = avg_rank_test
sub_rank.to_csv("submit_rank_ensemble.csv", header=None)
print(f"\n  submit_rank_ensemble.csv saved.")

# Submission: simple average (reference)
sub_simple = sample_submit.copy()
sub_simple[1] = avg_simple_test
sub_simple.to_csv("submit_simple_ensemble.csv", header=None)
print(f"  submit_simple_ensemble.csv saved.")

# Submission: voting (reference)
voting_model = make_tuned_voting()
voting_model.fit(X_tr_full, y)
pred_voting = voting_model.predict_proba(X_te_full)[:, 1]
sub_voting = sample_submit.copy()
sub_voting[1] = pred_voting
sub_voting.to_csv("submit_voting_reference.csv", header=None)
print(f"  submit_voting_reference.csv saved.")

# Compare submissions
print(f"\n  Submission prediction comparison:")
print(f"    Corr(rank, simple):  {np.corrcoef(avg_rank_test, avg_simple_test)[0, 1]:.6f}")
print(f"    Corr(rank, voting):  {np.corrcoef(avg_rank_test, pred_voting)[0, 1]:.6f}")
print(f"    Corr(simple, voting):{np.corrcoef(avg_simple_test, pred_voting)[0, 1]:.6f}")
print(f"    Max |rank - simple|: {np.max(np.abs(avg_rank_test - avg_simple_test)):.6f}")
print(f"    Max |rank - voting|: {np.max(np.abs(avg_rank_test - pred_voting)):.6f}")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

print(f"  Individual model AUCs:")
for name, auc in individual_aucs.items():
    print(f"    {name}: {auc:.4f}")

print(f"\n  Ensemble AUCs (fold-level CV):")
print(f"    Simple Average: {mean_simple:.4f}")
print(f"    Rank Average:   {mean_rank:.4f}")
print(f"    Equal Voting:   {mean_voting:.4f}")

best_method = max(
    [("Simple Average", mean_simple),
     ("Rank Average", mean_rank),
     ("Equal Voting", mean_voting)],
    key=lambda x: x[1],
)
print(f"\n  Best method: {best_method[0]} (AUC={best_method[1]:.4f})")

if best_method[0] == "Rank Average":
    print(f"  Rank averaging IMPROVED over simple average by {mean_rank - mean_simple:+.4f}")
    print(f"  Recommended submission: submit_rank_ensemble.csv")
elif best_method[0] == "Simple Average":
    print(f"  Simple averaging was best. Rank transform did not help.")
    print(f"  Recommended submission: submit_simple_ensemble.csv")
else:
    print(f"  Voting ensemble was best. Manual averaging did not help.")
    print(f"  Recommended submission: submit_voting_reference.csv")

print("\nDone!")
