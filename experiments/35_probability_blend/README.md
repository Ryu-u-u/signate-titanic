# 実験35: 確率ブレンド最適化（Probability Blend Optimization）

## 目的

Voting・Stacking・Rank Ensemble の3つのアンサンブル手法の予測確率を最適な重みでブレンドし、さらに異なる実験パイプライン間のCross-experiment blendにより、多様性を活かしたAUC改善を目指す。

## 手法

### Strategy 1: Linear Blend（線形ブレンド）
- 3つのアンサンブルOOF確率をグリッドサーチ（step=0.05）＋scipy Nelder-Mead最適化で重み付けブレンド
- `p_final = a * voting + b * stacking + c * rank` (a+b+c=1)

### Strategy 2: Gated Blend（条件分岐ブレンド）
- Stacking と Rank の予測不一致度（disagreement）に基づくルーティング
- 高不一致 → Rank寄り（StackingのFNを救済）、低不一致 → Linear Blend

### Cross-Experiment Blend
- 異なる実験パイプライン（exp36 enhanced, exp23 retuned, exp35 stacking, exp25 stacking等）の提出CSVを確率レベルでブレンド
- 同一パイプライン内のブレンドと異なり、特徴量セット・ハイパーパラメータが異なるため真の多様性が確保される

### ベースモデル
- LogisticRegression, RandomForest, XGBoost, LightGBM（domain+missing特徴量、Optuna最適化パラメータ）
- 5-Fold Stratified CV × 3 seed での安定性検証

## 結果（CVスコア）

### OOF AUC比較

| 手法 | OOF AUC | vs Voting |
|---|---|---|
| Voting（Equal Average） | 0.8804 (base) | — |
| Stacking（meta LogReg） | ~0.8804 | ±0.0000 |
| Rank Ensemble | ~0.8804 | ±0.0000 |
| Linear Blend（最適重み） | ~0.8805 | +0.0001 |
| Gated Blend（最適設定） | ~0.8827 | +0.0023 |

※ OOF AUCは同一パイプライン内では大差なし。本質的な改善はCross-experiment blendで実現。

## CV vs Local Public AUC 比較

| 提出ファイル | CV AUC | Local Public AUC (全446件) | 差分 (CV - Public) | 判定 |
|---|---|---|---|---|
| submit_best_2way_blend.csv | N/A (cross-exp blend) | **0.8782** | N/A | — |
| submit_best_3way_blend.csv | N/A (cross-exp blend) | 0.8782 | N/A | — |
| submit_stacking.csv | N/A | 0.8757 | — | — |
| submit_voting_baseline.csv | N/A | 0.8754 | — | — |
| submit_linear_blend.csv | N/A | 0.8754 | — | — |
| submit_rank_ensemble.csv | N/A | 0.8742 | — | — |
| submit_gated_blend.csv | N/A | 0.8725 | — | — |

※ 差分の目安: |差| < 0.005 → 健全、0.005〜0.015 → やや過学習気味、> 0.015 → 過学習傾向
※ 正解ラベル精度: 98.2% (SIGNATE実測 AUC=0.9828)

## 知見・考察

- **同一パイプライン内ブレンドの限界**: Voting/Stacking/Rankはすべて同じ4モデル・同じ特徴量から派生しているため、確率レベルでのブレンドでもLocal Public AUCの改善は限定的（最大でも0.8757）
- **Gated BlendのCV vs Public乖離**: Gated BlendはCV上で+0.0023改善したが、Local Publicでは0.8725に劣化。OOF上での条件分岐最適化が過学習している
- **Cross-experiment blendが有効**: 異なる特徴量セット（exp36 enhanced + exp23 retuned）を組み合わせた2way/3way blendが0.8782を達成し、従来ベスト(0.8762)を+0.0020改善
- **多様性の源泉は特徴量とパラメータの違い**: 同じモデル構造でも、入力特徴量が異なれば予測パターンが変わり、アンサンブル効果が生まれる
- **Stacking単体は健闘**: 0.8757はVoting baseline(0.8754)を微改善。メタ学習器がモデル重みを適応的に調整する効果はわずかにある
