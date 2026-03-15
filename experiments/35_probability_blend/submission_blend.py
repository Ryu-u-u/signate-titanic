"""Cross-Experiment Submission Blend.

Blend test predictions from different experiments that use different
feature sets and hyperparameters, providing genuine prediction diversity.

Key submissions:
  - exp36 enhanced voting:  0.8771 (exception features + prev params)
  - exp23 retuned voting:   0.8762 (domain+missing + Optuna-retuned params)
  - exp35 stacking:         0.8757 (meta-learner weighted, prev params)
  - exp25 stacking:         0.8748 (nested CV stacking)
"""
import sys
sys.path.append("../..")

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.config import SAMPLE_SUBMIT_CSV
from src.evaluation import evaluate_submission

# ============================================================
# Load Submissions
# ============================================================
print("=" * 60)
print("Cross-Experiment Submission Blend")
print("=" * 60)

submissions = {
    "enhanced_voting": "../36_hard_case_analysis/submit_enhanced_voting.csv",
    "retuned_voting": "../23_hp_retune_domain_missing/submit_retuned_voting.csv",
    "exp35_stacking": "submit_stacking.csv",
    "exp25_stacking": "../25_advanced_stacking/submit_stacking.csv",
    "exp26_single_seed": "../26_multi_seed/submit_single_seed_voting.csv",
    "exp28_rank_ens": "../28_rank_ensemble/submit_rank_ensemble.csv",
    "exp35_voting": "submit_voting_baseline.csv",
}

preds = {}
for name, path in submissions.items():
    try:
        df = pd.read_csv(path, header=None, names=["id", "prob"])
        preds[name] = df.set_index("id")["prob"]
        result = evaluate_submission(path)
        print(f"  {name}: Local AUC={result['auc']:.4f} (n={len(df)})")
    except Exception as e:
        print(f"  {name}: FAILED ({e})")

# ============================================================
# Pairwise Correlations
# ============================================================
print(f"\n  Pairwise Correlations:")
names = list(preds.keys())
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        corr = np.corrcoef(preds[names[i]], preds[names[j]])[0, 1]
        print(f"    {names[i]}-{names[j]}: {corr:.4f}")

# ============================================================
# Two-Way Blends
# ============================================================
print(f"\n" + "=" * 60)
print("Two-Way Blends (step=0.05)")
print("=" * 60)

sample_submit = pd.read_csv(SAMPLE_SUBMIT_CSV, index_col=0, header=None)
best_overall_auc = -1
best_overall_name = ""

# Focus on blending the top submissions
blend_pairs = [
    ("enhanced_voting", "retuned_voting"),
    ("enhanced_voting", "exp35_stacking"),
    ("enhanced_voting", "exp25_stacking"),
    ("retuned_voting", "exp35_stacking"),
    ("retuned_voting", "exp25_stacking"),
    ("enhanced_voting", "exp28_rank_ens"),
]

for name_a, name_b in blend_pairs:
    if name_a not in preds or name_b not in preds:
        continue

    best_w = 0.5
    best_auc = -1

    for w_int in range(0, 21):
        w = w_int * 0.05
        blend = w * preds[name_a] + (1 - w) * preds[name_b]

        sub = sample_submit.copy()
        sub[1] = blend.values
        sub.to_csv("_temp_blend.csv", header=None)
        result = evaluate_submission("_temp_blend.csv")

        if result["auc"] > best_auc:
            best_auc = result["auc"]
            best_w = w

    print(f"  {name_a} ({best_w:.2f}) + {name_b} ({1-best_w:.2f}): "
          f"AUC={best_auc:.4f}")

    if best_auc > best_overall_auc:
        best_overall_auc = best_auc
        best_overall_name = f"{name_a}({best_w:.2f})+{name_b}({1-best_w:.2f})"

        # Save best blend
        blend = best_w * preds[name_a] + (1 - best_w) * preds[name_b]
        sub = sample_submit.copy()
        sub[1] = blend.values
        sub.to_csv("submit_best_2way_blend.csv", header=None)

print(f"\n  Best 2-way: {best_overall_name}, AUC={best_overall_auc:.4f}")

# ============================================================
# Three-Way Blends
# ============================================================
print(f"\n" + "=" * 60)
print("Three-Way Blends")
print("=" * 60)

# Top 3 submissions for 3-way blend
top3_names = ["enhanced_voting", "retuned_voting", "exp35_stacking"]
if all(n in preds for n in top3_names):
    best_3way_auc = -1
    best_3way_weights = (0.33, 0.33, 0.34)

    step = 0.1
    for a_int in range(0, 11):
        a = a_int * step
        for b_int in range(0, 11 - a_int):
            b = b_int * step
            c = 1.0 - a - b
            if c < -1e-9:
                continue
            c = max(c, 0.0)

            blend = (a * preds[top3_names[0]]
                     + b * preds[top3_names[1]]
                     + c * preds[top3_names[2]])

            sub = sample_submit.copy()
            sub[1] = blend.values
            sub.to_csv("_temp_blend.csv", header=None)
            result = evaluate_submission("_temp_blend.csv")

            if result["auc"] > best_3way_auc:
                best_3way_auc = result["auc"]
                best_3way_weights = (a, b, c)

    a, b, c = best_3way_weights
    print(f"  {top3_names[0]} ({a:.1f}) + {top3_names[1]} ({b:.1f}) "
          f"+ {top3_names[2]} ({c:.1f}): AUC={best_3way_auc:.4f}")

    # Save
    blend = (a * preds[top3_names[0]]
             + b * preds[top3_names[1]]
             + c * preds[top3_names[2]])
    sub = sample_submit.copy()
    sub[1] = blend.values
    sub.to_csv("submit_best_3way_blend.csv", header=None)

# ============================================================
# Summary
# ============================================================
print(f"\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

final_submissions = [
    ("enhanced_voting (exp36)", "../36_hard_case_analysis/submit_enhanced_voting.csv"),
    ("retuned_voting (exp23)", "../23_hp_retune_domain_missing/submit_retuned_voting.csv"),
    ("best_2way_blend", "submit_best_2way_blend.csv"),
    ("best_3way_blend", "submit_best_3way_blend.csv"),
]

for name, path in final_submissions:
    try:
        result = evaluate_submission(path)
        print(f"  {name}: Local AUC={result['auc']:.4f}")
    except Exception as e:
        print(f"  {name}: FAILED ({e})")

# Cleanup temp file
import os
if os.path.exists("_temp_blend.csv"):
    os.remove("_temp_blend.csv")

print("\nDone!")
