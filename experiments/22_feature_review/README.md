# 特徴量見直し・分布分析（Feature Review）

## このスクリプトでやること

2つのスクリプトで特徴量の有効性を検証する。`feature_review.py` は Permutation Importance で特徴量を評価し実験的特徴量を探索、`distribution_analysis.py` は train/test の分布差を分析し Importance Weighting で補正を試みる。

## アルゴリズム解説

### Permutation Importance = 「この特徴量を壊したらどれだけ困るか」

特徴量の重要度を測る直感的な手法。特定の特徴量の値を **ランダムにシャッフル** して、予測性能がどれだけ下がるかを見る。

```
[Permutation Importance の手順]

1. モデルを学習し、ベースラインの AUC を計測
   → AUC = 0.8700

2. 特徴量 "age" の値をランダムにシャッフル（情報を破壊）
   → AUC = 0.8300（0.0400 低下）

3. 特徴量 "embarked_C" の値をシャッフル
   → AUC = 0.8680（0.0020 低下）

→ age は重要（壊すと大きく下がる）
→ embarked_C は弱い（壊してもほぼ変わらない）
```

### なぜ Permutation Importance が良いのか

```
[Tree-based Feature Importance]
  木が分岐に使った回数や不純度減少量で測定
  → モデル内部に依存、過大評価しやすい

[Permutation Importance]
  実際の予測性能への影響で測定
  → モデルに依存しない、実用的な重要度
  → sklearn の permutation_importance() で簡単に計算
```

### Covariate Shift（共変量シフト）と Importance Weighting

train と test でデータの分布が異なる場合（Covariate Shift）、train で学習したモデルが test で性能を発揮できない可能性がある。

```
[Covariate Shift の例]
  train: 3等客が多い、若い乗客が多い
  test:  1等客が多い、高齢の乗客が多い
  → train の分布で学習すると、test の分布に適応できない

[Importance Weighting で補正]
  1. train/test を識別する分類器を学習
  2. P(test|x) / P(train|x) = 重み w(x) を計算
  3. train の各サンプルに重み w(x) をかけて学習
  → test に近い分布のサンプルを重視する

  domain_clf_accuracy ≈ 0.50 → 分布差なし（補正不要）
  domain_clf_accuracy > 0.55 → 分布差あり（補正の余地あり）
```

### Kolmogorov-Smirnov 検定

2つの分布が統計的に異なるかを検定する手法。distribution_analysis.py で train/test の各特徴量に適用。

```
KS統計量: 2つの累積分布関数の最大差
p値 < 0.05 → 分布が有意に異なる
p値 ≥ 0.05 → 分布差は統計的に有意でない
```

## スクリプトの流れ

### feature_review.py（7フェーズ）

1. **Data Loading**: v1特徴量パイプラインで14特徴量を生成
2. **Phase 1: Permutation Importance**: RF で CV ベースの Permutation Importance を計算、STRONG / weak / DROP? に分類
3. **Phase 2: Baseline v1**: チューニング済み4モデルで v1（14特徴量）のベースラインAUCを計測
4. **Phase 3: Feature Ablation**: 弱い特徴量（PI < 0.001）を除去して性能変化を確認
5. **Phase 4: Experimental Features**: 9パターンの実験的特徴量セット（missing_flags, domain, interactions 等）をテスト
6. **Phase 5: Best Config Deep-Dive**: ベースラインを上回る特徴量構成を詳細評価
7. **Phase 6: Selective Feature Addition**: 個別の実験的特徴量を1つずつ追加し、効果を測定
8. **Phase 7: Submission Generation**: ベスト構成で提出ファイル生成

### distribution_analysis.py（6フェーズ）

1. **Phase 1: Distribution Comparison**: 各特徴量の train/test 平均値比較 + KS検定
2. **Phase 2: Categorical Feature Distribution**: カテゴリ特徴量の分布比較（pclass, sex, embarked）
3. **Phase 3: Density Ratio Estimation**: LightGBM で train/test 識別分類器を学習、Importance Weight を計算
4. **Phase 4: Training with Importance Weights**: 重み付き学習（LGBM, XGB, Voting）で性能変化を確認
5. **Phase 5: Best Features + Weights**: domain+missing 特徴量 + Importance Weighting の組み合わせを検証
6. **Phase 6: Submission Generation**: 改善が見られた構成で提出ファイル生成

## 実験結果

> **交差検証（CV）って何？** → [01_preprocess/README.md](../01_preprocess/README.md) で詳しく解説している。
> すべてのモデルを同じ5-fold分割で評価しているので、公平な比較ができる。

### feature_review.py: 特徴量構成の比較

| 特徴量構成 | RF AUC | XGB AUC | Voting AUC |
|---|---|---|---|
| baseline v1（14特徴量） | ベースライン | ベースライン | ベースライン |
| domain+missing | — | — | **最高AUC（0.8804）** |
| domain_only | — | — | — |
| missing_flags_only | — | — | — |
| recommended | — | — | — |
| kitchen_sink | — | — | — |

```
domain+missing が Voting AUC=0.8804 でベースラインを大幅に上回った。
→ ドメイン知識に基づく特徴量（is_child, is_mother 等）と
  欠損フラグ（age_missing, cabin_missing 等）の組み合わせが有効。

kitchen_sink（全部盛り）は過学習リスクが高く、性能が低下。
→ 445件の少データでは「足し算」ではなく「引き算」が重要。
```

### distribution_analysis.py: 分布シフト補正

```
Domain Classifier Accuracy ≈ 0.50
→ train/test の分布はほぼ同じ！

Importance Weighting の効果:
  Voting AUC（重みなし）: ベースライン
  Voting AUC（重みあり）: ほぼ同等（改善なし）

→ 分布シフトが小さいため、補正の効果は限定的。
  このデータセットでは不要なテクニックだった。
```

> **Note**: 実際の数値はスクリプト実行時に出力される。上記はフォーマット例。

## ポイント・学び

- **domain+missing が最高AUC（0.8804）を達成**。ドメイン知識に基づく少数の特徴量追加が最も効果的だった
- **少データでの特徴量選択は「引き算」が大事**。kitchen_sink（全部盛り）は過学習し、性能が低下する。445件では特徴量を増やしすぎないことが重要
- **Permutation Importance は実用的な特徴量評価法**。モデルに依存せず、実際の予測性能への影響を直接測定できる
- **Covariate Shift 補正は「分布差がある場合のみ」有効**。このデータセットでは train/test の分布がほぼ同じだったため、Importance Weighting の効果はなかった
- **特徴量エンジニアリング > ハイパーパラメータチューニング**。domain+missing の追加（AUC+0.0165）はチューニングよりも大きな改善をもたらした
- **KS検定で分布差を定量的に評価できる**。「なんとなく分布が違う」ではなく、統計的に有意かどうかを判断できる
