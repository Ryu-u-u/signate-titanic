# 実験28: Rank/Copula Ensemble（パーセンタイルランクアンサンブル）

## 目的

各モデルの予測確率をパーセンタイルランクに変換してから平均化することで、モデル間のスケール差を吸収し、より良いアンサンブルを実現する。

## 手法

### 使用モデル
- LogisticRegression, RandomForest, XGBoost, LightGBM（個別に学習）
- VotingClassifier (Soft Voting) は比較用リファレンス

### 特徴量
- domain + missing（ドメイン知識特徴量 + 欠損フラグ）
- `make_exp_builder(missing_flags=True, domain_features=True)` による構築

### アンサンブル手法の比較
1. **Simple Average**: 4モデルの予測確率をそのまま平均
2. **Rank Average**: 各モデルの予測確率を `rankdata(proba) / len(proba)` でパーセンタイルランクに変換後に平均
3. **Equal Voting (参考)**: sklearn VotingClassifier による Soft Voting

### ランク変換の仕組み
```
各モデルの予測確率 → scipy.stats.rankdata() → パーセンタイルランク (0~1)
→ 4モデル分のランクを平均化
```
- モデル間で予測確率のスケールが異なる場合（例: LogReg は 0.2~0.8、RF は 0.1~0.9）、ランク変換によりスケールが統一される

### 実験構成（4フェーズ）

1. **Phase 1: Out-of-Fold Predictions**
   - 4モデルの OOF 予測確率を取得（`cross_validate_oof` 使用）
   - 各モデルの個別 AUC と予測分布を確認

2. **Phase 2: Rank Transform and Average**
   - OOF 予測のランク変換と分布確認
   - Simple Average vs Rank Average の OOF AUC 比較
   - モデル間の予測相関行列（raw / rank-transformed）

3. **Phase 3: Fold-level Comparison**
   - 各fold で Simple Average / Rank Average / Equal Voting の AUC を比較
   - fold 平均 AUC と標準偏差の比較

4. **Phase 4: Submission Generation**
   - 全学習データで4モデルを学習 → テストデータの予測を生成
   - ランクアンサンブル / Simple Average / Voting の3種類の提出ファイルを生成
   - 3提出間の予測相関を確認

### 提出ファイル
- `submit_rank_ensemble.csv`: ランクアンサンブル（パーセンタイルランク平均）
- `submit_simple_ensemble.csv`: Simple Average（予測確率の単純平均）
- `submit_voting_reference.csv`: VotingClassifier の参考用

## 結果（CVスコア）

N/A（実行ログ未保存）

## CV vs Local Public AUC 比較

| 提出ファイル | CV AUC | Local Public AUC (全446件) | 差分 (CV - Public) | 判定 |
|---|---|---|---|---|
| submit_rank_ensemble.csv | N/A | 0.8742 | N/A | - |
| submit_simple_ensemble.csv | N/A | 0.8754 | N/A | - |
| submit_voting_reference.csv | N/A | 0.8754 | N/A | - |

**Local Public AUC 詳細:**

| 提出ファイル | 全446件 | sure 424件 | unique 359件 |
|---|---|---|---|
| submit_rank_ensemble.csv | 0.8742 | 0.8890 | 0.8868 |
| submit_simple_ensemble.csv | 0.8754 | 0.8911 | 0.8885 |
| submit_voting_reference.csv | 0.8754 | 0.8911 | 0.8885 |

※ 正解ラベル精度: 98.2% (SIGNATE実測 AUC=0.9828)

## 知見・考察

- **ランクアンサンブルは改善しなかった**: Rank Average (0.8742) < Simple Average (0.8754) = Voting (0.8754)
- ランク変換により -0.0012 の性能低下が発生した
- **Simple Average と Voting が同スコア (0.8754)**: sklearn VotingClassifier の Soft Voting は内部的に予測確率の平均化を行うため、個別学習+平均化と同等の結果になるのは理論通り
- **ランク変換が不利に働いた理由**: 4モデルの予測確率分布が既に似通っており（同じ特徴量・同じチューニング済みパラメータ）、スケール補正の必要性が低かった。むしろランク変換で情報が失われた可能性がある
- ランクアンサンブルが有効なのは、モデル間の予測スケールが大きく異なる場合（例: ニューラルネットと木モデルの組み合わせ）
