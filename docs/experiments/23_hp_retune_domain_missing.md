# 実験23: Domain+Missing 特徴量向け HP 再チューニング

## 目的

実験22で発見した最良特徴量セット（domain+missing, 約23特徴量）に対して、ハイパーパラメータを再チューニングする。
既存の BEST_PARAMS は v1（14特徴量）で最適化されたものであり、特徴量空間が拡大した domain+missing セットには最適でない可能性がある。
4モデル（LightGBM, XGBoost, RandomForest, LogisticRegression）それぞれを Optuna 100 trials で再チューニングし、改善を検証する。

## 手法

### 特徴量
- domain+missing 特徴量セット（約23特徴量）
  - ドメイン特徴量: is_child, is_mother 等
  - 欠損フラグ: age_missing, cabin_missing 等

### モデル
- LightGBM: num_leaves を 7-127 に拡大、colsample_bytree を 0.3-1.0 に拡大
- XGBoost: max_depth を 2-10 に拡大、colsample_bytree を 0.3-1.0 に拡大
- RandomForest: max_features に 0.3 を追加し、より広い探索範囲
- LogisticRegression: StandardScaler + Pipeline、C を 0.001-100.0 で探索

### チューニング
- Optuna TPESampler (seed固定) + MedianPruner
- 各モデル 100 trials
- 評価: 5-fold CV AUC

### アンサンブル
- Equal Voting (soft voting, 4モデル等重み)
- 重み最適化は Public スコアへの過学習リスクがあるため不採用

### 提出ファイル
1. `submit_retuned_voting.csv`: 再チューニングパラメータでの4モデル Voting
2. `submit_prev_params_voting.csv`: 旧パラメータ（v1チューニング）での4モデル Voting
3. `submit_retuned_lgbm.csv`: 再チューニング済み LightGBM 単体（ベスト単体モデル）

## 結果（CVスコア）

スクリプトは Optuna で動的にパラメータを探索し、CV AUC を出力する。
ベースラインの domain+missing Voting CV AUC は実験22で 0.8804 が報告されている。

各モデルの CV AUC はスクリプト実行時に以下の形式で出力される:
```
Phase 1: 各モデルの Best AUC（Optuna best_value）
Phase 2: 再チューニング後の個別モデル AUC + Voting AUC
Phase 3: 旧パラメータ vs 新パラメータの比較表
```

※ CV AUC の具体値はスクリプト実行ごとに Optuna の探索結果に依存するが、seed 固定により再現可能。

## CV vs Local Public AUC 比較

| 提出ファイル | CV AUC | Local Public AUC (全446件) | 差分 (CV - Public) | 判定 |
|---|---|---|---|---|
| submit_retuned_voting.csv | N/A | 0.8762 | N/A | - |
| submit_prev_params_voting.csv | N/A | 0.8754 | N/A | - |
| submit_retuned_lgbm.csv | N/A | 0.8754 | N/A | - |

※ CV AUC はスクリプト実行時に出力されるが、ログファイルとして保存されていないため N/A。
※ 差分の目安: |差| < 0.005 → 健全、0.005〜0.015 → やや過学習気味、> 0.015 → 過学習傾向
※ 正解ラベル精度: 98.2% (SIGNATE実測 AUC=0.9828)

## 知見・考察

- **再チューニング Voting が最高 Local Public AUC (0.8762)** を達成し、旧パラメータ Voting (0.8754) を +0.0008 上回った
- 旧パラメータ Voting と再チューニング LGBM 単体が同スコア (0.8754) であり、LGBM の改善が Voting 全体の改善に直結した可能性がある
- 特徴量空間の拡大（14→約23）に対して探索範囲を広げたことで、num_leaves や colsample_bytree の最適値がシフトした可能性がある
- 改善幅 (+0.0008) は小さく、v1 チューニングのパラメータでも domain+missing 特徴量に対して十分に汎化していたことを示唆する
- Equal Voting の採用は正しい判断。重み最適化は少数データ (445件) では Public スコアへの過学習リスクが高い
