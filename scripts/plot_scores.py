"""全提出CSVのスコアを可視化するスクリプト。"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
import numpy as np

# --- 日本語フォント設定 ---
jp_font = FontProperties(fname="/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf")
matplotlib.rcParams["font.family"] = "IPAGothic"
# rebuild font cache
matplotlib.font_manager._load_fontmanager(try_read_cache=False)

# --- データ ---
data = [
    ("36_hard_case\nenriched_retuned_blend", 0.8801, None),
    ("36_hard_case\nenriched_voting", 0.8787, 0.8807),
    ("35_prob_blend\nbest_2way_blend", 0.8782, None),
    ("35_prob_blend\nbest_3way_blend", 0.8782, None),
    ("36_hard_case\nenhanced_voting", 0.8771, 0.8817),
    ("36_hard_case\nenhanced_blend", 0.8771, None),
    ("23_hp_retune\nretuned_voting", 0.8762, None),
    ("35_prob_blend\nstacking", 0.8757, None),
    ("22_feature_review\ndomain_missing_voting", 0.8754, 0.8804),
    ("23_hp_retune\nprev_params_voting", 0.8754, None),
    ("23_hp_retune\nretuned_lgbm", 0.8754, None),
    ("26_multi_seed\nsingle_seed_voting", 0.8754, None),
    ("28_rank_ensemble\nsimple_ensemble", 0.8754, None),
    ("28_rank_ensemble\nvoting_reference", 0.8754, None),
    ("35_prob_blend\nvoting_baseline", 0.8754, None),
    ("35_prob_blend\nlinear_blend", 0.8754, None),
    ("36_hard_case\nbaseline_voting", 0.8754, 0.8804),
    ("26_multi_seed\nmulti_seed_voting", 0.8752, None),
    ("25_adv_stacking\nstacking", 0.8748, None),
    ("25_adv_stacking\nvoting_baseline", 0.8747, None),
    ("30_calibration\ncal_simple_avg", 0.8747, 0.8804),
    ("31_stability\nall_features_voting", 0.8747, 0.8804),
    ("32_pseudo_label\nbaseline", 0.8747, 0.8804),
    ("33_augmentation\nbaseline", 0.8747, 0.8804),
    ("34_bayesian\nbma_baseline", 0.8747, 0.8804),
    ("22_feature_review\nv1plus_voting", 0.8746, None),
    ("25_adv_stacking\nsklearn_stacking", 0.8745, None),
    ("28_rank_ensemble\nrank_ensemble", 0.8742, None),
    ("35_prob_blend\nrank_ensemble", 0.8742, None),
    ("30_calibration\ncal_logit_blend", 0.8742, 0.8804),
    ("31_stability\nstable_features", 0.8728, None),
    ("35_prob_blend\ngated_blend", 0.8725, None),
    ("21_tuning\nvoting_tuned", 0.8724, 0.8761),
    ("20_ensemble\nv1_voting", 0.8724, 0.8761),
    ("21_tuning\nvoting_weighted", 0.8720, 0.8795),
    ("12_xgb", 0.8707, 0.8561),
    ("21_tuning\nlgbm_tuned", 0.8705, 0.8786),
    ("20_ensemble\nsoft_voting", 0.8694, 0.8639),
    ("11_rf", 0.8683, 0.8603),
    ("22_feature_review\nweighted_voting", 0.8657, None),
    ("34_bayesian\nbma_reference", 0.8617, None),
    ("13_lgbm", 0.8574, 0.8429),
    ("14_nn", 0.8508, 0.7876),
    ("10_logreg", 0.8504, 0.8536),
    ("00_eda", 0.7010, None),
]

labels = [d[0] for d in data]
public_auc = [d[1] for d in data]
cv_auc = [d[2] for d in data]

# ================================================================
# Figure 1: Local Public AUC ランキング（横棒グラフ）
# ================================================================
fig1, ax1 = plt.subplots(figsize=(12, 18))

y_pos = np.arange(len(labels))
colors = []
for auc in public_auc:
    if auc >= 0.8770:
        colors.append("#2ecc71")
    elif auc >= 0.8754:
        colors.append("#3498db")
    elif auc >= 0.8740:
        colors.append("#f39c12")
    else:
        colors.append("#e74c3c")

bars = ax1.barh(y_pos, public_auc, color=colors, edgecolor="white", height=0.7)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels, fontsize=7, fontproperties=jp_font)
ax1.invert_yaxis()
ax1.set_xlabel("Local Public AUC", fontsize=12, fontproperties=jp_font)
ax1.set_title("Local Public AUC ランキング（全提出CSV）",
              fontsize=14, fontweight="bold", fontproperties=jp_font)
ax1.set_xlim(0.69, 0.886)

for i, (bar, auc) in enumerate(zip(bars, public_auc)):
    ax1.text(auc + 0.0005, bar.get_y() + bar.get_height() / 2,
             f"{auc:.4f}", va="center", fontsize=7, fontweight="bold")

ax1.axvline(x=0.8801, color="#2ecc71", linestyle="--", alpha=0.5, linewidth=1)
ax1.text(0.8801, -0.8, "Best: 0.8801", fontsize=8, color="#2ecc71", ha="center")

legend_elements = [
    Patch(facecolor="#2ecc71", label="Top tier (≧0.8770)"),
    Patch(facecolor="#3498db", label="Upper (0.8754–0.8769)"),
    Patch(facecolor="#f39c12", label="Mid (0.8740–0.8753)"),
    Patch(facecolor="#e74c3c", label="Lower (<0.8740)"),
]
ax1.legend(handles=legend_elements, loc="lower right", fontsize=9,
           prop=jp_font)

plt.tight_layout()
fig1.savefig("experiments/best/score_ranking.png", dpi=150, bbox_inches="tight")
print("Saved: experiments/best/score_ranking.png")

# ================================================================
# Figure 2: CV AUC vs Local Public AUC（散布図）
# ================================================================
fig2, ax2 = plt.subplots(figsize=(9, 8))

scatter_labels = []
scatter_cv = []
scatter_pub = []
for name, pub, cv in data:
    if cv is not None:
        scatter_labels.append(name.replace("\n", " / "))
        scatter_cv.append(cv)
        scatter_pub.append(pub)

ax2.scatter(scatter_cv, scatter_pub, s=80, c="#3498db", edgecolors="white",
            linewidth=1, zorder=5, alpha=0.85)

# y=x line
lims = [min(min(scatter_cv), min(scatter_pub)) - 0.005,
        max(max(scatter_cv), max(scatter_pub)) + 0.005]
ax2.plot(lims, lims, "--", color="gray", alpha=0.5, label="CV = Public (y=x)")
ax2.fill_between(lims, [l - 0.005 for l in lims], [l + 0.005 for l in lims],
                  alpha=0.08, color="green", label="|差分| < 0.005（健全）")

for i, label in enumerate(scatter_labels):
    short = label.split(" / ")[-1] if " / " in label else label
    offset_x, offset_y = 5, 5
    if scatter_cv[i] > 0.877:
        offset_y = -12 - (i % 3) * 10
    ax2.annotate(short, (scatter_cv[i], scatter_pub[i]),
                 textcoords="offset points", xytext=(offset_x, offset_y),
                 fontsize=6.5, alpha=0.8, fontproperties=jp_font)

ax2.set_xlabel("CV AUC", fontsize=12, fontproperties=jp_font)
ax2.set_ylabel("Local Public AUC", fontsize=12, fontproperties=jp_font)
ax2.set_title("CV AUC vs Local Public AUC（過学習チェック）",
              fontsize=14, fontweight="bold", fontproperties=jp_font)
ax2.legend(fontsize=9, loc="upper left", prop=jp_font)
ax2.set_aspect("equal")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig("experiments/best/cv_vs_public.png", dpi=150, bbox_inches="tight")
print("Saved: experiments/best/cv_vs_public.png")

# ================================================================
# Figure 3: 実験カテゴリ別ベストスコア比較
# ================================================================
fig3, ax3 = plt.subplots(figsize=(10, 6))

categories = [
    ("単体モデル\n(初期特徴量)", [
        ("LogReg", 0.8504), ("NN", 0.8508), ("LGBM", 0.8574),
        ("RF", 0.8683), ("XGB", 0.8707),
    ]),
    ("アンサンブル\n(初期)", [
        ("Soft Voting", 0.8694),
    ]),
    ("チューニング後\n(Optuna)", [
        ("Equal Voting", 0.8724), ("Weighted", 0.8720), ("LGBM tuned", 0.8705),
    ]),
    ("特徴量改善\n(domain+missing)", [
        ("domain Voting", 0.8754), ("retuned Voting", 0.8762),
    ]),
    ("高度手法", [
        ("Stacking", 0.8748), ("Rank Ens.", 0.8742),
        ("Calibration", 0.8747), ("Pseudo Label", 0.8747),
        ("Multi-seed", 0.8752), ("BMA", 0.8747),
    ]),
    ("確率ブレンド\n+ 外部データ強化", [
        ("Enhanced", 0.8771), ("Enriched", 0.8787),
        ("2way Blend", 0.8782), ("Enr+Ret Blend", 0.8801),
    ]),
]

x_offset = 0
x_ticks = []
x_tick_labels = []
cat_boundaries = []

for cat_name, models in categories:
    cat_start = x_offset
    for model_name, auc in models:
        color = "#2ecc71" if auc >= 0.8770 else "#3498db" if auc >= 0.8754 else "#f39c12" if auc >= 0.8740 else "#e74c3c"
        ax3.bar(x_offset, auc, width=0.7, color=color, edgecolor="white")
        ax3.text(x_offset, auc + 0.001, f"{auc:.4f}", ha="center", fontsize=6.5,
                 fontweight="bold", rotation=45)
        x_ticks.append(x_offset)
        x_tick_labels.append(model_name)
        x_offset += 1
    cat_end = x_offset - 1
    mid = (cat_start + cat_end) / 2
    ax3.text(mid, 0.838, cat_name, ha="center", fontsize=8, fontweight="bold",
             style="italic", color="#555", fontproperties=jp_font)
    if x_offset < sum(len(m) for _, m in categories):
        cat_boundaries.append(x_offset - 0.5)
    x_offset += 0.5

for b in cat_boundaries:
    ax3.axvline(x=b, color="gray", linestyle=":", alpha=0.4)

ax3.axhline(y=0.8801, color="#2ecc71", linestyle="--", alpha=0.4, linewidth=1)
ax3.text(x_offset - 1, 0.8805, "Top: 0.8801", fontsize=7, color="#2ecc71")

ax3.set_xticks(x_ticks)
ax3.set_xticklabels(x_tick_labels, fontsize=7, rotation=45, ha="right")
ax3.set_ylabel("Local Public AUC", fontsize=12, fontproperties=jp_font)
ax3.set_title("実験カテゴリ別スコア比較（各手法のベスト）",
              fontsize=14, fontweight="bold", fontproperties=jp_font)
ax3.set_ylim(0.835, 0.886)
ax3.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig3.savefig("experiments/best/score_by_category.png", dpi=150, bbox_inches="tight")
print("Saved: experiments/best/score_by_category.png")

print("\nDone! 3 charts generated.")
