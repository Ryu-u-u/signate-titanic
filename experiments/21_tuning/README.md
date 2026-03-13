# ハイパーパラメータチューニング（Optuna）

## このスクリプトでやること

Optuna（ベイズ最適化）で4モデルのハイパーパラメータを最適化し、さらに Weighted Voting の重みも最適化する。

## アルゴリズム解説

### Optuna = ベイズ最適化で「賢く探す」

グリッドサーチは全組み合わせを総当たりで試す。Optuna は **過去の試行結果を学習して、次に試すべきパラメータを賢く選ぶ**。

```
[グリッドサーチ]
  learning_rate: [0.01, 0.05, 0.1]
  max_depth:     [3, 5, 7]
  → 3 × 3 = 9通り全部試す（非効率）

[Optuna（TPE: Tree-structured Parzen Estimator）]
  Trial 1: lr=0.05, depth=5 → AUC=0.84
  Trial 2: lr=0.03, depth=3 → AUC=0.86  ← こっちが良い
  Trial 3: lr=0.02, depth=4 → AUC=0.87  ← 良い領域を重点的に探索
  ...
  Trial 100: 最適解に収束

→ 「良さそうな領域」を集中的に探索するので、少ない試行で最適解に近づける
```

### TPE（Tree-structured Parzen Estimator）の仕組み

```
1. 過去の試行を「良い結果」と「悪い結果」に分割
2. それぞれの分布を推定（Parzen Estimator）
3. 「良い結果の分布」から高確率でサンプリングされ、
   「悪い結果の分布」からは低確率のパラメータを次に選ぶ

→ 良い領域を優先的に探索しつつ、探索の多様性も保つ
```

### ゼロつく1との関連

ゼロつく1で学んだハイパーパラメータの手動調整を思い出してほしい：

```
学習率を変えて損失の推移を観察 → 良さそうな値を手動で選択
```

Optuna はこの「試行錯誤」を **自動化・効率化** したものだ。人間の勘と経験に頼る代わりに、数学的に最適な探索戦略を使う。

## チューニング対象

### 4モデル × パラメータ探索範囲

| モデル | パラメータ | 探索範囲 |
|---|---|---|
| **LightGBM** | n_estimators | 100〜1000 |
| | learning_rate | 0.01〜0.2（対数スケール） |
| | num_leaves | 7〜63 |
| | min_child_samples | 5〜50 |
| | subsample | 0.6〜1.0 |
| | colsample_bytree | 0.6〜1.0 |
| | reg_alpha / reg_lambda | 1e-8〜10.0（対数スケール） |
| **XGBoost** | n_estimators | 100〜1000 |
| | learning_rate | 0.01〜0.2（対数スケール） |
| | max_depth | 2〜8 |
| | subsample | 0.6〜1.0 |
| | colsample_bytree | 0.6〜1.0 |
| | reg_alpha / reg_lambda | 1e-8〜10.0（対数スケール） |
| | min_child_weight | 1〜10 |
| **RandomForest** | n_estimators | 100〜1000 |
| | max_depth | 3〜12 |
| | min_samples_leaf | 1〜10 |
| | min_samples_split | 2〜20 |
| | max_features | sqrt, log2, 0.5, 0.7 |
| **LogReg** | C | 0.001〜100.0（対数スケール） |
| | solver | liblinear, lbfgs |

## スクリプトの流れ

1. **Data Loading**: v1特徴量パイプラインで14特徴量を生成
2. **Phase 1: Optuna Hyperparameter Tuning**: 4モデル × 100 trials でベイズ最適化
3. **Phase 1.5: Verify Tuned Models**: チューニング済みパラメータでCV再評価
4. **Phase 2: Weighted Voting Optimization**: チューニング済みモデルで Weighted Voting の重み（4つ）を 50 trials で最適化
5. **Final Comparison**: ベースライン vs チューニング後のAUC比較テーブル
6. **Generate Submissions**: Equal Voting / Weighted Voting / ベスト単体モデルの3種類を提出

## 実験結果

> **交差検証（CV）って何？** → [01_preprocess/README.md](../01_preprocess/README.md) で詳しく解説している。
> すべてのモデルを同じ5-fold分割で評価しているので、公平な比較ができる。

### ベースライン vs チューニング後

| モデル | Baseline AUC | Tuned AUC | Diff |
|---|---|---|---|
| LogReg | 0.8536 | チューニング後 | 変化量 |
| RF | 0.8603 | チューニング後 | 変化量 |
| XGB | 0.8561 | チューニング後 | 変化量 |
| LGBM | 0.8429 | チューニング後 | 変化量 |
| Voting(equal, tuned) | 0.8639 | チューニング後 | 変化量 |
| Voting(weighted, tuned) | — | チューニング後 | vs Baseline Voting |

```
ベースライン: 20_ensemble の手動パラメータでの結果
チューニング後: Optuna 100 trials で最適化したパラメータでの結果

→ 各モデルのAUCがどれだけ改善したかを確認
→ Weighted Voting が Equal Voting を上回るかも検証
```

> **Note**: 実際の数値はスクリプト実行時に出力される。上記テーブルはフォーマット例。

## ポイント・学び

- **Optuna（TPE）は少ない試行で効率的に探索**。100 trials でグリッドサーチの何百通りもの組み合わせに匹敵する精度を達成できる
- **チューニングの効果はモデルによって異なる**。もともと手動パラメータが良かったモデル（RF）は改善幅が小さく、デフォルトから離れていたモデル（LGBM）は改善幅が大きい傾向がある
- **Weighted Voting vs Equal Voting**: 重みを最適化しても、445件の少量データでは Equal Voting との差は限定的。少データでは過学習リスクがあるため、シンプルな Equal が安定する場合も多い
- **チューニングは「最後の仕上げ」**。特徴量エンジニアリングやモデル選択の効果と比べると、ハイパーパラメータチューニングの改善幅は一般的に小さい
- **再現性の確保**: `TPESampler(seed=SEED)` で乱数シードを固定し、同じ結果を再現できるようにしている
