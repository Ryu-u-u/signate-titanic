# 実験29: CV-safe Target Encoding（ターゲットエンコーディング）

## 目的

低カーディナリティのカテゴリ組み合わせ（pclass x sex, embarked x sex, pclass x embarked）に対してターゲットエンコーディングを適用し、条件付き生存率を効率的に特徴量化する。

## 手法

### 使用モデル
- LogisticRegression, RandomForest, XGBoost, LightGBM
- VotingClassifier (Soft Voting) で最終アンサンブル
- 各モデルは BEST_PARAMS を使用

### 特徴量
- ベースライン: domain + missing（ドメイン知識特徴量 + 欠損フラグ）
- 追加: 3つの Target Encoding 特徴量
  - `te_pclass_sex`: pclass x sex の条件付き生存率
  - `te_embarked_sex`: embarked_S x sex の条件付き生存率
  - `te_pclass_embarked`: pclass x embarked_S の条件付き生存率

### Target Encoding の実装

**Leave-One-Out (LOO) + m-estimate smoothing:**
```
学習データ:
  encoded(x_i) = (group_sum - y_i + m * global_mean) / (group_count - 1 + m)
  → LOO: 自分自身を除外してターゲットリーク防止

テストデータ:
  encoded(x) = (group_sum + m * global_mean) / (group_count + m)
  → 全学習データの統計量を使用
```

- **m (smoothing parameter)**: 大きいほどグローバル平均に近づく（正則化が強い）
- **m値の探索**: m = 5, 10, 20, 50 の4パターンを評価

### CV-safe な設計
- Target Encoding の統計量は各CV fold の学習データのみから計算
- `make_te_builder(m)` がfold分割に対応した feature builder を返す
- テストデータへのリーク（target leakage）を完全に防止

### 実験構成（5フェーズ）

1. **Phase 1: LOO Target Encoding Implementation**
   - `loo_target_encode()` 関数の定義と動作確認

2. **Phase 2: Target-Encoding Feature Builder**
   - `make_te_builder(m)`: ベースの feature builder をラップし、TE特徴量を追加する builder を作成

3. **Phase 3: m-value Search**
   - m = 5, 10, 20, 50 の各値で全モデルのCV AUCを評価
   - Voting AUC が最大となる m を選択

4. **Phase 4: Compare vs Baseline**
   - ベースライン（domain+missing, TE なし）と最良 m での TE 追加を比較
   - 各モデル個別 + Voting での Delta を算出
   - m値感度テーブルの出力

5. **Phase 5: Generate Submission**
   - TE 追加で Voting AUC が改善した場合、TE 付きで提出ファイルを生成
   - ベースライン提出ファイルも参考用に生成

### 提出ファイル
- なし（submit CSV 未生成）

## 結果（CVスコア）

N/A（実行ログ未保存）

※ submit CSV が存在しないため、Target Encoding がベースラインを上回らなかった可能性が高い

## 知見・考察

- **Target Encoding のCV-safe実装**: LOO + m-estimate smoothing により、ターゲットリークを防止しつつカテゴリ間の条件付き確率を特徴量化した
- **submit CSV が未生成**: Phase 5 の条件分岐で TE がベースラインを上回らなかった可能性が高い
- **445件の小データでの限界**: Target Encoding は大規模データで効果を発揮しやすいが、少データでは LOO 後のサンプル数が少なく、エンコード値のノイズが大きくなる
- **pclass x sex は既にドメイン特徴量でカバー**: ベースラインの domain features（is_child, is_mother, family_size 等）が既に pclass や sex との交互作用を暗黙的に表現しており、追加のTE特徴量は冗長だった可能性がある
- **m値の選択**: m が小さいとグループ固有の情報を重視（過学習リスク大）、m が大きいとグローバル平均に近づく（特徴量としての情報量低下）。小データではこのトレードオフが厳しい
