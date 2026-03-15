# 実験24: CatBoost + 5モデル Voting

## 目的

CatBoost を既存の4モデルアンサンブルに追加し、5モデル Equal Voting で精度が向上するかを検証する。
CatBoost の Ordered Boosting はアンサンブルの多様性（diversity）を高め、既存の GBDT 系モデル（LightGBM, XGBoost）とは異なる予測パターンを生み出す可能性がある。

## 手法

### 特徴量
- domain+missing 特徴量セット（約23特徴量）
  - ドメイン特徴量: is_child, is_mother 等
  - 欠損フラグ: age_missing, cabin_missing 等

### フェーズ構成

**Phase 1: CatBoost ベースライン**
- デフォルトパラメータ (iterations=500, learning_rate=0.05, depth=6)
- 5-fold CV で AUC, Accuracy, F1, LogLoss を計測

**Phase 2: Optuna HP チューニング (100 trials)**
- 探索範囲:
  - iterations: 100-1000
  - learning_rate: 0.01-0.2 (log)
  - depth: 3-8
  - l2_leaf_reg: 1e-8 - 10.0 (log)
  - min_data_in_leaf: 1-50
  - subsample: 0.6-1.0 (Bernoulli bootstrap)
- TPESampler (seed固定) + MedianPruner

**Phase 3: 4モデル vs 5モデル Voting 比較**
- 4モデル Voting: LogReg + RF + XGB + LGBM (既存ベスト)
- 5モデル Voting: LogReg + RF + XGB + LGBM + CatBoost (tuned)
- 全7構成（各単体 + 2種 Voting）の AUC を比較

**Phase 4: 提出ファイル生成**
- `submission_5model_voting.csv`: 5モデル Voting
- `submission_catboost_tuned.csv`: チューニング済み CatBoost 単体

### モデル (既存4モデル)
- LogisticRegression (StandardScaler + Pipeline)
- RandomForest
- XGBoost (GPU: device="cuda")
- LightGBM

## 結果（CVスコア）

CV AUC はスクリプト実行時に出力される。以下の情報が出力される:

```
Phase 1: CatBoost (default) AUC, Accuracy, F1, LogLoss + per-fold AUC
Phase 2: CatBoost (tuned) Best AUC + パラメータ + verification結果
Phase 3: 全7構成の AUC, Accuracy 一覧 + Voting AUC change
```

ベースラインの4モデル Voting CV AUC は実験22で 0.8804 が報告されている。

※ submit CSV は生成されていない（実験結果として CV 比較のみ実施）。

## 知見・考察

- CatBoost の Ordered Boosting は理論的にアンサンブルの多様性を高めるが、445件の小規模データセットではその効果が限定的である可能性がある
- 5モデル Voting で改善が見られるかどうかは、CatBoost が既存の GBDT 系モデル（LGBM, XGB）と十分に異なる予測を生成できるかに依存する
- 提出 CSV が生成されていないことから、5モデル Voting が4モデル Voting を有意に上回らなかった可能性が示唆される
- CatBoost の depth 探索範囲 (3-8) は小規模データに適切な設定であり、過学習抑制を意識した設計
- GPU 利用は XGBoost のみ (device="cuda")。CatBoost は CPU で実行されている
