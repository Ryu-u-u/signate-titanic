<div class="hero-banner" markdown>

## SIGNATE タイタニック生存予測

**Public AUC 0.8828** 達成 — 36実験の軌跡

[ベスト結果を見る :material-arrow-right:](results.md){ .md-button .md-button--primary }

</div>

!!! success "ベストスコア: Public AUC 0.8828"
    外部データ特徴量強化 + Cross-experiment Blend で達成。
    Local推定 AUC 0.8801 に対して +0.0027 の上振れ。

[SIGNATE タイタニック号の生存予測](https://signate.jp/competitions/102) コンペティション用プロジェクト。

「ゼロから作るDeep Learning」を読了した初学者が、古典的な機械学習アルゴリズムを体系的に学ぶための実験記録。

## 実験結果サマリー

### モデル別 CV スコア（5-fold StratifiedKFold）

| # | モデル | AUC | Accuracy | 備考 |
|---|--------|-----|----------|------|
| 1 | **Voting (domain+missing)** | **0.8804** | 0.8090 | v1 + ドメイン特徴量 + 欠損フラグ |
| 2 | Voting (v1, tuned) | 0.8761 | 0.8000 | Optuna チューニング済み 4モデル |
| 3 | LightGBM (tuned) | 0.8786 | — | Optuna 100 trials |
| 4 | XGBoost (tuned) | 0.8727 | — | Optuna 100 trials |
| 5 | RandomForest (tuned) | 0.8698 | — | Optuna 100 trials |
| 6 | Voting (baseline) | 0.8639 | 0.8000 | デフォルトパラメータ 4モデル |
| 7 | Stacking | 0.8636 | 0.8022 | メタ学習器: LogReg |
| 8 | RandomForest | 0.8603 | 0.8045 | デフォルトパラメータ |
| 9 | XGBoost | 0.8561 | 0.8000 | デフォルトパラメータ |
| 10 | LogisticRegression | 0.8536 | 0.8000 | デフォルトパラメータ |
| 11 | LightGBM | 0.8429 | 0.7865 | デフォルトパラメータ |
| 12 | Neural Network (MLP) | 0.7876 | 0.7596 | 小データで不安定 |

### Public スコア（提出結果）

| モデル | Public AUC |
|--------|-----------|
| Voting (tuned, equal weight) | **0.8757** |
| Voting (tuned, weighted) | 0.8755 |
| Voting (baseline) | 0.8718 |
| RF (baseline) | 0.8697 |
| LGBM (tuned) | 0.8626 |

## プロジェクト構成

```
├── src/                    共通ライブラリ
│   ├── config.py           パス・定数管理
│   ├── data.py             データ読み込み
│   ├── features.py         特徴量エンジニアリング (v0/v1)
│   ├── exp_features.py     実験用特徴量カタログ (8カテゴリ)
│   ├── evaluation.py       CV評価パイプライン
│   └── utils.py            seed固定等
├── experiments/            実験ノートブック
│   ├── 00_eda/             探索的データ分析
│   ├── 01_preprocess/      前処理比較 (v0 vs v1)
│   ├── 02_feature_lab/     特徴量エンジニアリング実験室
│   ├── 10_logreg/          Logistic Regression
│   ├── 11_rf/              Random Forest
│   ├── 12_xgb/             XGBoost
│   ├── 13_lgbm/            LightGBM
│   ├── 14_nn/              Neural Network (MLP)
│   ├── 20_ensemble/        アンサンブル (Voting / Stacking)
│   ├── 21_tuning/          Optuna ハイパーパラメータチューニング
│   ├── 22_feature_review/  特徴量見直し・分布分析
│   └── best/               ベスト結果まとめ
├── scripts/                CLIスクリプト
│   └── evaluate_submission.py  ローカルPublic AUCシミュレーション
├── configs/
│   └── default.yaml        ベースラインハイパーパラメータ
├── data/
│   ├── raw/                SIGNATE 公式データ (gitignore)
│   └── external/           外部データソース (gitignore)
├── pyproject.toml          依存関係 (uv)
└── requirements.txt        依存関係 (pip)
```

## 実験の流れ

### 1. EDA（探索的データ分析）
`experiments/00_eda/` — データの全体像を把握。sex, pclass, fare が生存に強く影響。

### 2. 前処理
`experiments/01_preprocess/` — v0（基本）vs v1（leak-safe）を比較。pclass 別中央値補完で AUC 改善。

### 3. 個別モデル
`experiments/10_logreg/` ~ `14_nn/` — 各アルゴリズムを単体で評価。RF が単体最強（AUC=0.8603）。

### 4. アンサンブル
`experiments/20_ensemble/` — Soft Voting が全モデル中最高スコア（AUC=0.8639）を記録。

### 5. ハイパーパラメータチューニング
`experiments/21_tuning/` — Optuna で 4 モデル × 100 trials。Voting AUC 0.8639 → 0.8761。

### 6. 特徴量見直し
`experiments/22_feature_review/` — Permutation Importance 分析 + ドメイン特徴量追加。Voting AUC 0.8761 → 0.8804。

## 主な知見

- **Permutation Importance**: sex, age, fare_per_person の 3 特徴量が支配的。fare, log_fare は冗長
- **少データ（445件）では**: 特徴量の盛りすぎ（kitchen_sink）は過学習で悪化。厳選が有効
- **アンサンブルの力**: 異なるアルゴリズムの Soft Voting で単体モデルを安定的に上回る
- **CV と Public の乖離**: Optuna チューニング後は CV > Public の傾向。過適合に注意
- **分布シフト補正**: train/test の分布差はほぼゼロ（KS 検定全特徴 p>0.5）。重み付けは不発

## 特徴量バージョン

| Version | 特徴量数 | 内容 |
|---------|---------|------|
| v0 | 9 | 基本前処理 (median 補完 + sex エンコード + embarked one-hot) |
| v1 | 14 | v0 + family_size, is_alone, log_fare, fare_per_person, pclass_sex |
| v1+domain+missing | ~23 | v1 + is_child, is_mother, fare_zero, family bins + 欠損フラグ |
