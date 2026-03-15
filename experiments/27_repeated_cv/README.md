# 実験27: Repeated-CV Robust Tuning（安定性重視のハイパーパラメータ再チューニング）

## 目的

通常のCV AUC最大化ではなく、`mean(AUC) - lambda * std(AUC)` というロバスト目的関数でOptunaチューニングを行い、CV-to-Public のギャップ（過学習）を低減する。

## 手法

### 使用モデル
- LogisticRegression, RandomForest, XGBoost, LightGBM
- VotingClassifier (Soft Voting) で最終アンサンブル

### 特徴量
- domain + missing（ドメイン知識特徴量 + 欠損フラグ）
- `make_exp_builder(missing_flags=True, domain_features=True)` による構築

### ロバスト評価関数
```
robust_score = mean(AUC) - lambda * std(AUC)
```
- **CV_SEEDS**: [42, 123, 456, 789, 2024] の5シードでCV分割
- **5シード x 5fold = 25fold** の全AUCから mean / std を算出
- **lambda = 0.5**: 安定性と性能のバランス

### 実験構成（4フェーズ）

1. **Phase 1: Baseline Robust Evaluation**
   - 既存 BEST_PARAMS での各モデル（LogReg, RF, XGB, LGBM, Voting）のロバストスコアを計測

2. **Phase 2: Optuna Robust Re-tuning**
   - 各モデルについて N_TRIALS=50 で Optuna 最適化
   - 目的関数は `mean(AUC) - 0.5 * std(AUC)`
   - 1トライアルあたり 5シード x 5fold = 25回のCV評価（非常に高コスト）

3. **Phase 3: Compare Robust-tuned vs BEST_PARAMS**
   - ロバストチューニング後の各モデルと BEST_PARAMS を比較
   - Voting アンサンブルでの最終比較

4. **Phase 4: Generate Submission**
   - Voting のロバストスコアが改善した場合のみ、ロバストチューニングパラメータで提出ファイルを生成
   - 改善しなかった場合は BEST_PARAMS を使用

### 提出ファイル
- なし（submit CSV 未生成）

## 結果（CVスコア）

N/A（実行ログ未保存）

※ submit CSV が存在しないため、ロバストチューニングが BEST_PARAMS を上回らなかった、もしくは実行途中で中断された可能性がある

## 知見・考察

- **ロバスト目的関数 `mean - lambda * std` の設計思想**: 単純なAUC最大化は fold 間の分散が大きいパラメータを選びやすく、Public スコアとの乖離（過学習）に繋がる。stdにペナルティを課すことで安定性の高いパラメータを探索する
- **計算コストが非常に高い**: 1トライアルあたり25回のCV評価が必要で、4モデル x 50トライアル = 5000回のCV評価が必要
- **submit CSV が未生成**: Phase 4 の条件分岐で BEST_PARAMS Voting が勝った（ロバスト再チューニングの効果がなかった）可能性が高い
- 445件の小データでは、チューニングの探索空間が限られており、BEST_PARAMS（既にOptuna 200トライアルで最適化済み）から大きな改善は難しい
- **lambda = 0.5 の妥当性**: stdへのペナルティが強すぎると性能を犠牲にしすぎ、弱すぎると安定化効果がない。0.5 は中間的な設定だが、データセットに応じた調整が必要
