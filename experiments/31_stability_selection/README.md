# 実験31: Stability Feature Selection（安定特徴量選択）

## 目的

複数のシード・フォールドにわたって一貫して重要な特徴量だけを選択（Boruta的アプローチ）することで、ノイズ特徴量を除去し汎化性能を向上させるかを検証する。

## 手法

- **ベースモデル**: RandomForest（Permutation Importance計算用）、評価は全4モデル + Voting
- **特徴量**: domain+missing（make_exp_builder）
- **安定性評価**:
  - 5シード（42, 123, 456, 789, 2024） x 5フォールド = 25回評価
  - 各評価で Permutation Importance（n_repeats=10, scoring=roc_auc）を計算
  - PI > 0.001 の回数をカウント
  - 25回中15回以上（60%以上）で重要とされた特徴量を「安定特徴量」として選定
- **比較**:
  - 全特徴量 vs 安定特徴量のみ
  - 全5モデル（LogReg, RF, XGB, LGBM, Voting）で評価

### 処理フロー

1. **Phase 1**: 5 seeds x 5 folds = 25回のPermutation Importance計算
2. **Phase 2**: 安定特徴量の選定（60%閾値）
3. **Phase 3**: 全特徴量 vs 安定特徴量でCV比較
4. **Phase 4**: 提出ファイル生成（2種類: stable features, all features）

## 結果（CVスコア）

| 構成 | Voting CV AUC | 備考 |
|---|---|---|
| 全特徴量 | 0.8804 | ベースライン |
| 安定特徴量のみ | スクリプト出力参照 | 60%閾値で選定 |

※ 安定特徴量のリストと除外された特徴量はスクリプト出力で確認可能。

## CV vs Local Public AUC 比較

| 提出ファイル | CV AUC | Local Public AUC (全446件) | 差分 (CV - Public) | 判定 |
|---|---|---|---|---|
| submit_stable_features_voting.csv | N/A (安定特徴量 Voting) | 0.8728 | N/A | - |
| submit_all_features_voting.csv | 0.8804 | 0.8747 | +0.0057 | やや過学習気味 |

※ 差分の目安: |差| < 0.005 → 健全、0.005〜0.015 → やや過学習気味、> 0.015 → 過学習傾向
※ 正解ラベル精度: 98.2% (SIGNATE実測 AUC=0.9828)

## 知見・考察

- **安定特徴量のみ（0.8728）は全特徴量（0.8747）を下回った**。特徴量の除外が有効に働かず、むしろ情報量の減少が性能低下を招いた
- **全特徴量のVoting（0.8747）がLocal Public AUCでも安定**。CV-Public差は+0.0057でやや過学習気味だが許容範囲
- **Permutation Importanceの閾値（0.001）や安定性閾値（60%）の設定が結果に影響**。閾値を厳しくしすぎると有用な特徴量まで除外するリスクがある
- **445件の小規模データでは、tree-basedモデルの内部正則化（min_child_samples, max_depth等）が既にノイズ特徴量を無視する効果を持つ**ため、外部からの特徴量選択の恩恵が限定的
- **安定特徴量の情報は、他実験での特徴量理解には有用**。どの特徴量が安定して重要かのリストは、ドメイン理解の参考になる
