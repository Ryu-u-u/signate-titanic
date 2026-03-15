"""Experiment 30: Calibration-First Ensembling.

Hypothesis: Calibrating each model's probabilities before blending
improves ensemble quality by aligning output scales across diverse models.

Phase 1: Evaluate calibration of each base model (Brier score, calibration curve)
Phase 2: CalibratedClassifierCV with Platt (sigmoid) and isotonic methods
Phase 3: Logit-space blending of calibrated OOF probabilities
Phase 4: Generate submission if improved
Phase 5: Final comparison table
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
from scipy.special import logit, expit
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import TRAIN_CSV, TEST_CSV, SAMPLE_SUBMIT_CSV, SEED, N_FOLDS, TARGET_COL
from src.utils import seed_everything
from src.features import compute_train_stats, build_pipeline
from src.exp_features import make_exp_builder
from src.evaluation import cross_validate, cross_validate_oof, calc_metrics

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
# Feature Builder: domain+missing
# ============================================================
fb = make_exp_builder(
    missing_flags=True, age_bins=None, fare_bins=None,
    interactions=False, polynomial=False, group_stats=False,
    freq_encoding=False, rank_features=False, domain_features=True,
)

# ============================================================
# BEST_PARAMS
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


# ============================================================
# Model Factory Functions
# ============================================================
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
    return XGBClassifier(**p, random_state=SEED, eval_metric="logloss", device="cuda")


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
        ], voting="soft",
    )


MODEL_FNS = {
    "LogReg": make_tuned_logreg,
    "RF": make_tuned_rf,
    "XGB": make_tuned_xgb,
    "LGBM": make_tuned_lgbm,
}

# ============================================================
# Phase 1: Evaluate Calibration of Each Base Model
# ============================================================
print("=" * 60)
print("Phase 1: Base Model Calibration Diagnostics")
print("=" * 60)

base_oof_probas = {}
base_metrics = {}

for name, model_fn in MODEL_FNS.items():
    fold_m, mean_m, oof_pred, oof_proba, models = cross_validate_oof(
        model_fn, X_raw, y, feature_builder=fb,
    )
    base_oof_probas[name] = oof_proba
    base_metrics[name] = mean_m

    # Brier score loss
    brier = brier_score_loss(y, oof_proba)

    # Calibration curve (10 bins)
    frac_pos, mean_pred = calibration_curve(y, oof_proba, n_bins=10, strategy="uniform")

    print(f"\n  {name}:")
    print(f"    AUC={mean_m['auc']:.4f}, Brier={brier:.4f}")
    print(f"    Calibration curve (mean_pred -> actual_frac):")
    for mp, fp in zip(mean_pred, frac_pos):
        bar = "#" * int(fp * 40)
        print(f"      {mp:.3f} -> {fp:.3f}  |{bar}")

# Equal Voting baseline (simple average of OOF probas)
avg_proba = np.mean(list(base_oof_probas.values()), axis=0)
avg_auc = roc_auc_score(y, avg_proba)
avg_brier = brier_score_loss(y, avg_proba)
print(f"\n  SimpleAvg Blend: AUC={avg_auc:.4f}, Brier={avg_brier:.4f}")

# Voting classifier baseline
_, voting_mean_m = cross_validate(make_tuned_voting, X_raw, y, feature_builder=fb)
print(f"  VotingClassifier: AUC={voting_mean_m['auc']:.4f}")


# ============================================================
# Phase 2: CalibratedClassifierCV (Platt & Isotonic)
# ============================================================
print("\n" + "=" * 60)
print("Phase 2: Calibrated Models (Platt sigmoid & Isotonic)")
print("=" * 60)


def make_calibrated_model(base_fn, method="sigmoid"):
    """Wrap a model factory with CalibratedClassifierCV."""
    def factory():
        return CalibratedClassifierCV(base_fn(), method=method, cv=3)
    return factory


cal_results = {}

for method in ["sigmoid", "isotonic"]:
    print(f"\n--- Calibration method: {method} ---")
    for name, model_fn in MODEL_FNS.items():
        cal_fn = make_calibrated_model(model_fn, method=method)
        _, mean_m = cross_validate(cal_fn, X_raw, y, feature_builder=fb)
        key = f"{name}_{method}"
        cal_results[key] = mean_m
        diff = mean_m["auc"] - base_metrics[name]["auc"]
        print(f"  {name} ({method}): AUC={mean_m['auc']:.4f} (vs base {base_metrics[name]['auc']:.4f}, diff={diff:+.4f})")


# ============================================================
# Phase 3: Logit-Space Blending
# ============================================================
print("\n" + "=" * 60)
print("Phase 3: Logit-Space Blending of Calibrated OOF Probabilities")
print("=" * 60)

# Determine best calibration method per model
best_cal_method = {}
for name in MODEL_FNS:
    sig_auc = cal_results[f"{name}_sigmoid"]["auc"]
    iso_auc = cal_results[f"{name}_isotonic"]["auc"]
    base_auc = base_metrics[name]["auc"]
    best_method = None
    best_auc = base_auc
    if sig_auc > best_auc:
        best_method = "sigmoid"
        best_auc = sig_auc
    if iso_auc > best_auc:
        best_method = "isotonic"
        best_auc = iso_auc
    best_cal_method[name] = best_method
    print(f"  {name}: best calibration = {best_method or 'none (keep base)'} (AUC={best_auc:.4f})")

# Get OOF predictions from calibrated (or base) models
cal_oof_probas = {}
print("\n  Getting OOF predictions for calibrated/best models...")
for name, model_fn in MODEL_FNS.items():
    method = best_cal_method[name]
    if method is not None:
        fn = make_calibrated_model(model_fn, method=method)
    else:
        fn = model_fn
    _, _, _, oof_proba, _ = cross_validate_oof(fn, X_raw, y, feature_builder=fb)
    cal_oof_probas[name] = oof_proba
    print(f"    {name}: mean_proba={oof_proba.mean():.4f}, std={oof_proba.std():.4f}")

# Simple average of calibrated probabilities
cal_avg_proba = np.mean(list(cal_oof_probas.values()), axis=0)
cal_avg_auc = roc_auc_score(y, cal_avg_proba)
cal_avg_brier = brier_score_loss(y, cal_avg_proba)
print(f"\n  Calibrated SimpleAvg: AUC={cal_avg_auc:.4f}, Brier={cal_avg_brier:.4f}")

# Logit-space blending
clipped = {name: np.clip(proba, 0.001, 0.999)
           for name, proba in cal_oof_probas.items()}
logits = {name: logit(p) for name, p in clipped.items()}
avg_logit = np.mean(list(logits.values()), axis=0)
blend_proba = expit(avg_logit)

logit_auc = roc_auc_score(y, blend_proba)
logit_brier = brier_score_loss(y, blend_proba)
print(f"  Logit-Space Blend:    AUC={logit_auc:.4f}, Brier={logit_brier:.4f}")

# Uncalibrated simple average for comparison
uncal_avg_auc = roc_auc_score(y, avg_proba)
uncal_avg_brier = brier_score_loss(y, avg_proba)

# Uncalibrated logit blend
uncal_clipped = {name: np.clip(proba, 0.001, 0.999)
                 for name, proba in base_oof_probas.items()}
uncal_logits = {name: logit(p) for name, p in uncal_clipped.items()}
uncal_avg_logit = np.mean(list(uncal_logits.values()), axis=0)
uncal_blend_proba = expit(uncal_avg_logit)
uncal_logit_auc = roc_auc_score(y, uncal_blend_proba)
uncal_logit_brier = brier_score_loss(y, uncal_blend_proba)

print(f"\n  --- Comparison ---")
print(f"  Uncal SimpleAvg:    AUC={uncal_avg_auc:.4f}, Brier={uncal_avg_brier:.4f}")
print(f"  Uncal Logit Blend:  AUC={uncal_logit_auc:.4f}, Brier={uncal_logit_brier:.4f}")
print(f"  Cal SimpleAvg:      AUC={cal_avg_auc:.4f}, Brier={cal_avg_brier:.4f}")
print(f"  Cal Logit Blend:    AUC={logit_auc:.4f}, Brier={logit_brier:.4f}")
print(f"  VotingClassifier:   AUC={voting_mean_m['auc']:.4f}")


# ============================================================
# Phase 4: Generate Submission if Improved
# ============================================================
print("\n" + "=" * 60)
print("Phase 4: Submission Generation")
print("=" * 60)

sample_submit = pd.read_csv(SAMPLE_SUBMIT_CSV, index_col=0, header=None)

# Determine best blending approach
blend_options = {
    "uncal_simple_avg": uncal_avg_auc,
    "uncal_logit_blend": uncal_logit_auc,
    "cal_simple_avg": cal_avg_auc,
    "cal_logit_blend": logit_auc,
    "voting_classifier": voting_mean_m["auc"],
}
best_blend_name = max(blend_options, key=blend_options.get)
best_blend_auc = blend_options[best_blend_name]
print(f"  Best blending: {best_blend_name} (AUC={best_blend_auc:.4f})")

# Train final models on full data and predict test
X_tr_full, X_te_full = fb(X_raw, test)
print(f"  Full train features: {X_tr_full.shape}")
print(f"  Full test features:  {X_te_full.shape}")

# Generate submission with best approach
test_probas = {}
for name, model_fn in MODEL_FNS.items():
    method = best_cal_method[name]
    if method is not None:
        fn = make_calibrated_model(model_fn, method=method)
    else:
        fn = model_fn
    model = fn()
    model.fit(X_tr_full, y)
    proba = model.predict_proba(X_te_full)[:, 1]
    test_probas[name] = proba

# Calibrated logit blend submission
test_clipped = {name: np.clip(p, 0.001, 0.999) for name, p in test_probas.items()}
test_logits = {name: logit(p) for name, p in test_clipped.items()}
test_avg_logit = np.mean(list(test_logits.values()), axis=0)
test_blend = expit(test_avg_logit)

sub_logit = sample_submit.copy()
sub_logit[1] = test_blend
sub_logit.to_csv("submit_cal_logit_blend.csv", header=None)
print(f"  submit_cal_logit_blend.csv: mean={test_blend.mean():.3f}, std={test_blend.std():.3f}")

# Calibrated simple average submission
test_cal_avg = np.mean(list(test_probas.values()), axis=0)
sub_cal_avg = sample_submit.copy()
sub_cal_avg[1] = test_cal_avg
sub_cal_avg.to_csv("submit_cal_simple_avg.csv", header=None)
print(f"  submit_cal_simple_avg.csv: mean={test_cal_avg.mean():.3f}, std={test_cal_avg.std():.3f}")

# Voting classifier submission (baseline)
voting_model = make_tuned_voting()
voting_model.fit(X_tr_full, y)
voting_pred = voting_model.predict_proba(X_te_full)[:, 1]
sub_voting = sample_submit.copy()
sub_voting[1] = voting_pred
sub_voting.to_csv("submit_voting_baseline.csv", header=None)
print(f"  submit_voting_baseline.csv: mean={voting_pred.mean():.3f}, std={voting_pred.std():.3f}")


# ============================================================
# Phase 5: Final Comparison Table
# ============================================================
print("\n" + "=" * 60)
print("Phase 5: Final Comparison Table")
print("=" * 60)

print(f"\n  {'Method':<30} {'AUC':>8} {'Brier':>8} {'vs Voting':>10}")
print("  " + "-" * 60)

rows = [
    ("Uncal SimpleAvg", uncal_avg_auc, uncal_avg_brier),
    ("Uncal Logit Blend", uncal_logit_auc, uncal_logit_brier),
    ("Cal SimpleAvg", cal_avg_auc, cal_avg_brier),
    ("Cal Logit Blend", logit_auc, logit_brier),
    ("VotingClassifier", voting_mean_m["auc"], None),
]

voting_auc = voting_mean_m["auc"]
for label, auc, brier in rows:
    diff = auc - voting_auc
    brier_str = f"{brier:.4f}" if brier is not None else "  N/A "
    marker = " ***" if diff > 0.001 else ""
    print(f"  {label:<30} {auc:>8.4f} {brier_str:>8} {diff:>+10.4f}{marker}")

print(f"\n  Individual model AUCs (base vs best-calibrated):")
print(f"  {'Model':<12} {'Base AUC':>10} {'Best Cal':>10} {'Method':>10} {'Diff':>8}")
print("  " + "-" * 54)
for name in MODEL_FNS:
    base_auc = base_metrics[name]["auc"]
    method = best_cal_method[name]
    if method is not None:
        cal_key = f"{name}_{method}"
        cal_auc = cal_results[cal_key]["auc"]
    else:
        cal_auc = base_auc
    diff = cal_auc - base_auc
    method_str = method if method else "none"
    print(f"  {name:<12} {base_auc:>10.4f} {cal_auc:>10.4f} {method_str:>10} {diff:>+8.4f}")

print("\nDone!")
