# 実験26: Multi-Seed Averaging（マルチシード平均化）

## 目的

複数のランダムシード（42, 123, 456, 789, 2024）でモデルを学習し、予測確率を平均化することで分散を低減し、汎化性能の安定化を図る。

## 手法

### 使用モデル
- **VotingClassifier (Soft Voting)**: LogisticRegression + RandomForest + XGBoost + LightGBM
- 各モデルは実験23で得られた BEST_PARAMS を使用

### 特徴量
- domain + missing（ドメイン知識特徴量 + 欠損フラグ）
- `make_exp_builder(missing_flags=True, domain_features=True)` による構築

### 実験構成（3フェーズ）

1. **Phase 1: Individual Seed Evaluation**
   - 5つのシードそれぞれで Voting モデルのCV AUCを評価
   - シード間の分散（std, range）を計測

2. **Phase 2: Multi-Seed Average CV Evaluation**
   - 各foldで5シード分のモデルを学習し、予測確率を平均化
   - Single-seed (seed=42) との比較を fold 単位で実施
   - Phase 2.5: 個別モデル（LogReg, RF, XGB, LGBM, Voting）ごとのシード間分散を計測

3. **Phase 3: Submission Generation**
   - 全学習データで5シード分のモデルを学習 → 予測確率を平均化して提出ファイルを生成
   - Single-seed (seed=42) の提出ファイルも参考用に生成
   - 両者の予測相関と最大差分を出力

### 提出ファイル
- `submit_multi_seed_voting.csv`: 5シード平均化した予測確率
- `submit_single_seed_voting.csv`: seed=42 のみの予測確率（参考用）

## 結果（CVスコア）

N/A（実行ログ未保存）

※ ベースラインの domain+missing Voting CV AUC は実験22時点で 0.8804

## CV vs Local Public AUC 比較

| 提出ファイル | CV AUC | Local Public AUC (全446件) | 差分 (CV - Public) | 判定 |
|---|---|---|---|---|
| submit_multi_seed_voting.csv | N/A | 0.8752 | N/A | - |
| submit_single_seed_voting.csv | N/A | 0.8754 | N/A | - |

**Local Public AUC 詳細:**

| 提出ファイル | 全446件 | sure 424件 | unique 359件 |
|---|---|---|---|
| submit_multi_seed_voting.csv | 0.8752 | 0.8909 | 0.8880 |
| submit_single_seed_voting.csv | 0.8754 | 0.8911 | 0.8885 |

※ 正解ラベル精度: 98.2% (SIGNATE実測 AUC=0.9828)

## 知見・考察

- **Multi-seed averaging は Public AUC を改善しなかった**: single-seed (0.8754) > multi-seed (0.8752) で差は -0.0002
- 5シード平均化の分散低減効果はあるはずだが、446件のテストデータでは差が出にくい
- BEST_PARAMS の Voting 自体が既に4モデルのアンサンブルであるため、シード平均化による追加の多様性獲得効果が限定的だった可能性がある
- single-seed の方が微差で上回っており、マルチシード平均化は今回のデータセットでは有効ではなかった
