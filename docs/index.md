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

> **Best Score: SIGNATE Public AUC 0.8828** — 外部データ特徴量強化 + Cross-experiment blend（exp36）
>
> 📖 [実験記録ドキュメントサイト](https://ryu-u-u.github.io/signate-titanic/)

## 実験結果サマリー

### Public スコア（SIGNATE 提出結果）

| # | モデル | Public AUC | 手法 |
|---|--------|-----------|------|
| 1 | **Enriched + Retuned Blend** | **0.8828** | 外部データ特徴量 + Cross-experiment blend (exp36) |
| 2 | Enriched Voting | — | Title/Deck/Ticket 特徴量 (exp36) |
| 3 | Best 2-way/3-way Blend | — | Cross-experiment blend (exp35) |
| 4 | Voting (tuned, equal weight) | 0.8757 | Optuna チューニング (exp21) |
| 5 | Voting (tuned, weighted) | 0.8755 | 重み付き Voting (exp21) |
| 6 | Voting (baseline) | 0.8718 | デフォルトパラメータ (exp20) |

> 2位以下の SIGNATE 提出スコアは未取得（Local Public AUC による推定順位）。詳細は `experiments/best/README.md` 参照。

### モデル別 CV スコア（5-fold StratifiedKFold）

| # | モデル | AUC | Accuracy | 備考 |
|---|--------|-----|----------|------|
| 1 | **Enriched Voting** | **0.8807** | — | 外部データ特徴量 (exp36) |
| 2 | Voting (domain+missing) | 0.8804 | 0.8090 | v1 + ドメイン特徴量 + 欠損フラグ (exp22) |
| 3 | Voting (v1, tuned) | 0.8761 | 0.8000 | Optuna チューニング済み 4モデル (exp21) |
| 4 | LightGBM (tuned) | 0.8786 | — | Optuna 100 trials (exp21) |
| 5 | XGBoost (tuned) | 0.8727 | — | Optuna 100 trials (exp21) |
| 6 | RandomForest (tuned) | 0.8698 | — | Optuna 100 trials (exp21) |
| 7 | Voting (baseline) | 0.8639 | 0.8000 | デフォルトパラメータ 4モデル (exp20) |
| 8 | Stacking | 0.8636 | 0.8022 | メタ学習器: LogReg (exp20) |
| 9 | RandomForest | 0.8603 | 0.8045 | デフォルトパラメータ (exp11) |
| 10 | XGBoost | 0.8561 | 0.8000 | デフォルトパラメータ (exp12) |
| 11 | LogisticRegression | 0.8536 | 0.8000 | デフォルトパラメータ (exp10) |
| 12 | LightGBM | 0.8429 | 0.7865 | デフォルトパラメータ (exp13) |
| 13 | Neural Network (MLP) | 0.7876 | 0.7596 | 小データで不安定 (exp14) |

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
│   ├── 23_hp_retune_domain_missing/  HP再チューニング (domain+missing)
│   ├── 24_catboost/        CatBoost + 5モデル Voting
│   ├── 25_advanced_stacking/  Advanced Stacking (Nested CV)
│   ├── 26_multi_seed/      Multi-Seed Averaging
│   ├── 27_repeated_cv/     Repeated-CV Robust Tuning
│   ├── 28_rank_ensemble/   Rank/Copula Ensemble
│   ├── 29_target_encoding/ CV-safe Target Encoding
│   ├── 30_calibration/     Calibration-First Ensembling
│   ├── 31_stability_selection/  Stability Feature Selection
│   ├── 32_pseudo_labeling/ Agreement-Gated Pseudo Labeling
│   ├── 33_augmentation/    Tabular Augmentation (Mixup)
│   ├── 34_bayesian/        Bayesian Model Averaging
│   ├── 35_probability_blend/  確率ブレンド最適化
│   ├── 36_hard_case_analysis/ Hard Case分析 + 外部データ特徴量強化
│   └── best/               ベスト結果まとめ・全提出ランキング
├── scripts/                CLIスクリプト
│   ├── evaluate_submission.py  ローカルPublic AUCシミュレーション
│   ├── build_docs.py       ドキュメント自動生成
│   ├── analyze_predictions.py  予測分析
│   └── plot_scores.py      スコア可視化
├── docs/                   GitHub Pages ドキュメントサイト
├── mkdocs.yml              MkDocs Material 設定
├── .github/workflows/      CI/CD (GitHub Pages デプロイ)
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

### 7. 高度アンサンブル探索
`experiments/23_hp_retune_domain_missing/` ~ `28_rank_ensemble/` — CatBoost追加、Advanced Stacking、Multi-Seed、Repeated-CV、Rank Ensemble 等を検証。Equal Voting を有意に超える手法は見つからず。

### 8. 高度手法検証
`experiments/29_target_encoding/` ~ `34_bayesian/` — Target Encoding、Calibration、Stability Selection、Pseudo Labeling、Mixup、BMA を検証。445件の小規模データでは高度手法の効果は限定的。

### 9. 最終最適化（ベストスコア達成）
`experiments/35_probability_blend/` — Cross-experiment blend で異なるパイプライン間の予測を最適ブレンド。
`experiments/36_hard_case_analysis/` — Hard Case 分析 → 外部データから Title/Deck/Ticket 特徴量を復元 → Enriched + Retuned Blend で **Public AUC 0.8828** を達成。

## 主な知見

- **外部データ特徴量が最大のレバー**: Title（敬称）、Cabin Deck、Ticket Group Size を外部データから復元し、最大の改善を実現（exp36）
- **Cross-experiment blend の有効性**: 異なる特徴量セット・パラメータの提出 CSV をブレンドすることで、同一パイプライン内ブレンドより大きく改善（exp35）
- **データ拡張・高度手法は効果なし**: Pseudo Labeling, Mixup, BMA, Calibration 等は 445件の小規模データでは改善なし（exp29-34）
- **Permutation Importance**: sex, age, fare_per_person の 3 特徴量が支配的。fare, log_fare は冗長
- **少データ（445件）では**: 特徴量の盛りすぎ（kitchen_sink）は過学習で悪化。厳選が有効
- **アンサンブルの力**: 異なるアルゴリズムの Soft Voting で単体モデルを安定的に上回る
- **CV と Public の乖離**: Optuna チューニング後は CV > Public の傾向。過適合に注意
- **Equal Voting が堅牢**: 重み付き・Stacking・Rank アンサンブルのいずれも Equal Voting を同一特徴量で有意に上回れず

## 特徴量バージョン

| Version | 特徴量数 | 内容 |
|---------|---------|------|
| v0 | 9 | 基本前処理 (median 補完 + sex エンコード + embarked one-hot) |
| v1 | 14 | v0 + family_size, is_alone, log_fare, fare_per_person, pclass_sex |
| v1+domain+missing | ~23 | v1 + is_child, is_mother, fare_zero, family bins + 欠損フラグ |
| enriched | ~30 | v1+domain+missing + title, cabin_deck, ticket_group_size (外部データ復元) |
