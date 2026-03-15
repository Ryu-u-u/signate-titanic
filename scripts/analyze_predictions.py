"""レコード単位の正誤分析スクリプト。

各モデルがどのレコードを正解/不正解したかを分析し、
難易度マップ・モデル相補性・最適組み合わせの判断材料を提供する。
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- フォント設定 ---
jp_font = FontProperties(fname="/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf")
matplotlib.rcParams["font.family"] = "IPAGothic"
matplotlib.font_manager._load_fontmanager(try_read_cache=False)

OUT_DIR = Path("experiments/best")

# --- データ読み込み ---
gt = pd.read_csv("data/external/test_ground_truth.csv")
gt = gt.set_index("id")

# 分析対象モデル（代表的なもの。冗長な重複提出は除外）
models = {
    # 単体モデル
    "LogReg": "experiments/10_logreg/submit.csv",
    "RF": "experiments/11_rf/submit.csv",
    "XGB": "experiments/12_xgb/submit.csv",
    "LGBM": "experiments/13_lgbm/submit.csv",
    "NN": "experiments/14_nn/submit.csv",
    # アンサンブル系
    "Voting_init": "experiments/20_ensemble/submit.csv",
    "Voting_tuned": "submit_voting_tuned.csv",
    "Weighted_tuned": "submit_voting_weighted.csv",
    "LGBM_tuned": "submit_lgbm_tuned.csv",
    # 特徴量改善
    "Domain_voting": "submit_domain_missing_voting.csv",
    "Retuned_voting": "experiments/23_hp_retune_domain_missing/submit_retuned_voting.csv",
    "Retuned_lgbm": "experiments/23_hp_retune_domain_missing/submit_retuned_lgbm.csv",
    # 高度手法
    "Stacking": "experiments/25_advanced_stacking/submit_stacking.csv",
    "Rank_ens": "experiments/28_rank_ensemble/submit_rank_ensemble.csv",
    "Stable_feat": "experiments/31_stability_selection/submit_stable_features_voting.csv",
    "BMA_ref": "experiments/34_bayesian/submit_bma_reference.csv",
    # 確率ブレンド + 外部データ強化 (exp35/36)
    "Enhanced_vot": "experiments/36_hard_case_analysis/submit_enhanced_voting.csv",
    "Enriched_vot": "experiments/36_hard_case_analysis/submit_enriched_voting.csv",
    "Enriched_blend": "experiments/36_hard_case_analysis/submit_enriched_retuned_blend.csv",
    "Linear_blend": "experiments/35_probability_blend/submit_linear_blend.csv",
    "2way_blend": "experiments/35_probability_blend/submit_best_2way_blend.csv",
}

# 予測確率と二値予測を読み込み
probs = {}
preds = {}
for name, path in models.items():
    df = pd.read_csv(path, header=None, names=["id", "prob"])
    df = df.set_index("id")
    probs[name] = df["prob"]
    preds[name] = (df["prob"] >= 0.5).astype(int)

prob_df = pd.DataFrame(probs)
pred_df = pd.DataFrame(preds)

# ground truth と揃える
common_ids = gt.index.intersection(pred_df.index)
gt_aligned = gt.loc[common_ids]
pred_aligned = pred_df.loc[common_ids]
prob_aligned = prob_df.loc[common_ids]

y_true = gt_aligned["survived"]
n_samples = len(common_ids)
n_models = len(models)

# --- 正誤マトリクス ---
correct_matrix = pred_aligned.eq(y_true, axis=0).astype(int)

# レコードごとの正解モデル数
correct_count = correct_matrix.sum(axis=1)
# モデルごとの正解数
model_accuracy = correct_matrix.mean(axis=0).sort_values(ascending=False)

print(f"=== レコード単位 正誤分析 ===")
print(f"対象レコード数: {n_samples}")
print(f"分析モデル数: {n_models}")
print()

# ================================================================
# 1. モデル別正解率
# ================================================================
print("--- モデル別正解率 (閾値=0.5) ---")
for name, acc in model_accuracy.items():
    n_correct = correct_matrix[name].sum()
    print(f"  {name:20s}: {acc:.4f} ({n_correct}/{n_samples})")
print()

# ================================================================
# 2. レコード難易度分布
# ================================================================
print("--- レコード難易度分布 ---")
for i in range(n_models + 1):
    count = (correct_count == i).sum()
    if count > 0:
        pct = count / n_samples * 100
        label = "全モデル不正解" if i == 0 else f"{i}モデル正解" if i < n_models else "全モデル正解"
        print(f"  {label:20s}: {count:3d}件 ({pct:.1f}%)")
print()

# 難しいレコード（半数以上のモデルが不正解）
hard_threshold = n_models // 2
hard_records = correct_count[correct_count <= hard_threshold].sort_values()
print(f"--- 難しいレコード（{hard_threshold}モデル以下が正解） ---")
print(f"  計 {len(hard_records)}件")

# 全モデル不正解のレコード
all_wrong = correct_count[correct_count == 0]
print(f"\n--- 全モデル不正解のレコード ({len(all_wrong)}件) ---")
if len(all_wrong) > 0:
    for idx in all_wrong.index:
        true_label = y_true.loc[idx]
        avg_prob = prob_aligned.loc[idx].mean()
        conf = gt_aligned.loc[idx, "confidence"]
        print(f"  id={idx}: 正解={true_label}, 平均予測確率={avg_prob:.4f}, "
              f"信頼度={conf}")
print()

# 1モデルだけ正解のレコード
one_correct = correct_count[correct_count == 1]
print(f"--- 1モデルだけ正解のレコード ({len(one_correct)}件) ---")
for idx in one_correct.index[:20]:
    true_label = y_true.loc[idx]
    winner = correct_matrix.loc[idx]
    winner_name = winner[winner == 1].index[0]
    avg_prob = prob_aligned.loc[idx].mean()
    conf = gt_aligned.loc[idx, "confidence"]
    print(f"  id={idx}: 正解={true_label}, 唯一正解={winner_name}, "
          f"平均予測確率={avg_prob:.4f}, 信頼度={conf}")
if len(one_correct) > 20:
    print(f"  ... 他 {len(one_correct) - 20}件")
print()

# ================================================================
# 3. モデル間の相補性分析
# ================================================================
print("--- モデル間の誤答重複率 ---")
# 各モデルペアで「両方不正解」の割合
error_matrix = 1 - correct_matrix
model_names = list(models.keys())

# 相補性スコア: モデルAの誤答のうち、モデルBが正解する割合
comp_scores = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
for a in model_names:
    for b in model_names:
        if a == b:
            comp_scores.loc[a, b] = 0.0
        else:
            a_errors = error_matrix[a] == 1
            if a_errors.sum() == 0:
                comp_scores.loc[a, b] = 1.0
            else:
                comp_scores.loc[a, b] = correct_matrix.loc[a_errors, b].mean()

print("相補性スコア = モデルAの誤答のうち、モデルBが正解する割合")
print("（高いほど相補的）")
print()

# 各モデルの最良パートナーを表示
for a in model_names:
    best_partner = comp_scores.loc[a].astype(float).drop(a).idxmax()
    score = comp_scores.loc[a, best_partner]
    print(f"  {a:20s} → ベストパートナー: {best_partner} ({score:.3f})")
print()

# ================================================================
# 4. ペアワイズ組み合わせ分析
# ================================================================
print("--- ペアワイズ OR 正解率（2モデル併用時の正解率上限） ---")
pair_results = []
for i, a in enumerate(model_names):
    for j, b in enumerate(model_names):
        if i < j:
            # どちらか一方でも正解なら正解
            or_correct = ((correct_matrix[a] == 1) | (correct_matrix[b] == 1)).mean()
            pair_results.append((a, b, or_correct))

pair_results.sort(key=lambda x: -x[2])
print("Top 10 ペア:")
for a, b, score in pair_results[:10]:
    print(f"  {a:20s} + {b:20s}: {score:.4f}")
print()

# ================================================================
# 5. 正解パターンによるレコードクラスタリング
# ================================================================
# レコードを正解パターンでグループ化
pattern_groups = correct_matrix.apply(lambda row: tuple(row), axis=1)
pattern_counts = pattern_groups.value_counts()
print(f"--- ユニークな正解パターン数: {len(pattern_counts)} ---")
print("Top 10 パターン:")
for pattern, count in pattern_counts.head(10).items():
    n_correct_in_pattern = sum(pattern)
    models_wrong = [m for m, c in zip(model_names, pattern) if c == 0]
    if len(models_wrong) <= 3:
        wrong_str = ", ".join(models_wrong) if models_wrong else "なし"
    else:
        wrong_str = f"{len(models_wrong)}モデル不正解"
    print(f"  {count:3d}件: {n_correct_in_pattern}/{n_models}正解 "
          f"(不正解: {wrong_str})")
print()

# ================================================================
# 6. 信頼度別の分析
# ================================================================
print("--- 信頼度別の平均正解率 ---")
for conf in ["unique", "sure", "all"]:
    if conf == "all":
        mask = gt_aligned.index
    elif conf == "sure":
        mask = gt_aligned[gt_aligned["confidence"].isin(["unique", "all_agree"])].index
    else:
        mask = gt_aligned[gt_aligned["confidence"] == conf].index
    if len(mask) > 0:
        avg_acc = correct_matrix.loc[mask].mean()
        overall = avg_acc.mean()
        print(f"  {conf:10s} ({len(mask):3d}件): 全モデル平均正解率 = {overall:.4f}")
print()

# ================================================================
# 可視化
# ================================================================

# --- Fig 1: レコード難易度ヒストグラム ---
fig1, ax1 = plt.subplots(figsize=(10, 5))
bins = np.arange(-0.5, n_models + 1.5, 1)
ax1.hist(correct_count, bins=bins, color="#3498db", edgecolor="white", alpha=0.85)
ax1.set_xlabel("正解モデル数", fontsize=12, fontproperties=jp_font)
ax1.set_ylabel("レコード数", fontsize=12, fontproperties=jp_font)
ax1.set_title("レコード難易度分布（正解モデル数別）",
              fontsize=14, fontweight="bold", fontproperties=jp_font)
ax1.set_xticks(range(n_models + 1))
ax1.axvline(x=hard_threshold, color="red", linestyle="--", alpha=0.5)
ax1.text(hard_threshold - 0.3, ax1.get_ylim()[1] * 0.9, "← 難", fontsize=10,
         color="red", fontproperties=jp_font)
ax1.text(hard_threshold + 0.3, ax1.get_ylim()[1] * 0.9, "易 →", fontsize=10,
         color="green", fontproperties=jp_font)
ax1.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig1.savefig(OUT_DIR / "record_difficulty.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'record_difficulty.png'}")

# --- Fig 2: モデル間相補性ヒートマップ ---
fig2, ax2 = plt.subplots(figsize=(14, 12))
comp_float = comp_scores.astype(float)
mask_diag = np.eye(len(model_names), dtype=bool)
sns.heatmap(comp_float, annot=True, fmt=".2f", cmap="YlOrRd",
            mask=mask_diag, ax=ax2, vmin=0, vmax=1,
            xticklabels=model_names, yticklabels=model_names,
            cbar_kws={"label": "相補性スコア"})
ax2.set_title("モデル間相補性（行モデルの誤答を列モデルが正解する割合）",
              fontsize=13, fontweight="bold", fontproperties=jp_font)
ax2.set_xlabel("列: 救済モデル", fontsize=11, fontproperties=jp_font)
ax2.set_ylabel("行: 誤答モデル", fontsize=11, fontproperties=jp_font)
plt.tight_layout()
fig2.savefig(OUT_DIR / "model_complementarity.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'model_complementarity.png'}")

# --- Fig 3: モデル別正解率 + 難しいレコードの内訳 ---
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))

# 左: モデル別正解率
sorted_acc = model_accuracy.sort_values()
colors3 = ["#2ecc71" if v > 0.80 else "#f39c12" if v > 0.78 else "#e74c3c"
            for v in sorted_acc.values]
ax3a.barh(range(len(sorted_acc)), sorted_acc.values, color=colors3, edgecolor="white")
ax3a.set_yticks(range(len(sorted_acc)))
ax3a.set_yticklabels(sorted_acc.index, fontsize=9)
ax3a.set_xlabel("正解率 (閾値=0.5)", fontsize=11, fontproperties=jp_font)
ax3a.set_title("モデル別正解率", fontsize=13, fontweight="bold", fontproperties=jp_font)
for i, v in enumerate(sorted_acc.values):
    ax3a.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8)
ax3a.grid(axis="x", alpha=0.3)

# 右: 正解ラベル別の誤答分析（0を1と予測 vs 1を0と予測）
fp_counts = []  # False Positive: 0を1と予測
fn_counts = []  # False Negative: 1を0と予測
for name in model_names:
    fp = ((pred_aligned[name] == 1) & (y_true == 0)).sum()
    fn = ((pred_aligned[name] == 0) & (y_true == 1)).sum()
    fp_counts.append(fp)
    fn_counts.append(fn)

x_pos = np.arange(n_models)
width = 0.35
ax3b.bar(x_pos - width / 2, fp_counts, width, label="FP (0→1と誤予測)",
         color="#e74c3c", alpha=0.8)
ax3b.bar(x_pos + width / 2, fn_counts, width, label="FN (1→0と誤予測)",
         color="#3498db", alpha=0.8)
ax3b.set_xticks(x_pos)
ax3b.set_xticklabels(model_names, fontsize=6, rotation=60, ha="right")
ax3b.set_ylabel("件数", fontsize=11, fontproperties=jp_font)
ax3b.set_title("誤答タイプ内訳 (FP vs FN)", fontsize=13,
               fontweight="bold", fontproperties=jp_font)
ax3b.legend(prop=jp_font, fontsize=9)
ax3b.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig3.savefig(OUT_DIR / "model_accuracy_detail.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'model_accuracy_detail.png'}")

# --- Fig 4: 難しいレコードの予測確率ヒートマップ（Top 30） ---
hardest_ids = correct_count.sort_values().head(30).index
hard_probs = prob_aligned.loc[hardest_ids]
hard_true = y_true.loc[hardest_ids]
hard_correct_n = correct_count.loc[hardest_ids]

fig4, ax4 = plt.subplots(figsize=(14, 10))
# ラベルにid, 正解, 正解モデル数を表示
row_labels = [f"id={idx} (正解={hard_true.loc[idx]}, {hard_correct_n.loc[idx]}/{n_models}正解)"
              for idx in hardest_ids]

sns.heatmap(hard_probs.values, annot=True, fmt=".2f", cmap="RdYlGn",
            xticklabels=model_names, yticklabels=row_labels,
            ax=ax4, vmin=0, vmax=1, cbar_kws={"label": "予測確率"})
ax4.set_title("難しいレコード Top30 の各モデル予測確率",
              fontsize=14, fontweight="bold", fontproperties=jp_font)
ax4.set_xlabel("モデル", fontsize=11, fontproperties=jp_font)

# 正解ラベルに応じて行ラベルに色付け
for i, label in enumerate(ax4.get_yticklabels()):
    label.set_fontproperties(jp_font)
    label.set_fontsize(7)
    if hard_true.iloc[i] == 1:
        label.set_color("#2ecc71")
    else:
        label.set_color("#e74c3c")

plt.tight_layout()
fig4.savefig(OUT_DIR / "hard_records_heatmap.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'hard_records_heatmap.png'}")

# --- CSVエクスポート: レコード×モデル正誤マトリクス ---
export_df = correct_matrix.copy()
export_df.insert(0, "survived", y_true)
export_df.insert(1, "confidence", gt_aligned["confidence"])
export_df["correct_count"] = correct_count
export_df["avg_prob"] = prob_aligned.mean(axis=1).round(4)
export_df = export_df.sort_values("correct_count")
export_df.to_csv(OUT_DIR / "record_model_matrix.csv")
print(f"Saved: {OUT_DIR / 'record_model_matrix.csv'}")

print("\nDone!")
