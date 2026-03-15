# 実験33: Tabular Augmentation（Mixupベースデータ拡張）

## 目的

Mixupによるテーブルデータの合成サンプル生成で、445件の小規模訓練データを拡張し性能向上を図る。同一クラス内のサンプルペアを内挿することで、決定境界の滑らかさ向上を期待する。

## 手法

- **ベースモデル**: LogReg, RandomForest, XGBoost, LightGBM, Voting（チューニング済み）
- **特徴量**: domain+missing（make_exp_builder）
- **Mixup手法**:
  - 同一クラス内でランダムにペアを選び、内挿で合成サンプルを生成
  - `x_new = lam * x_i + (1 - lam) * x_j`（lam ~ Beta(alpha, alpha)）
  - クラスラベルは親サンプルのラベルを保持（同一クラス内Mixupのため）
- **探索パラメータ**:
  - n_aug（クラスあたりの合成数）: 50, 100, 200
  - alpha（Beta分布パラメータ、小さいほど原サンプルに近い）: 0.1, 0.2, 0.4

### 処理フロー

1. **Phase 0**: ベースラインCV（拡張なし）
2. **Phase 1**: 合成サンプルの統計量分析（元データとの分布比較）
3. **Phase 2**: n_aug を変えてCV評価（alpha=0.2固定）
4. **Phase 2b**: alpha を変えてCV評価（n_aug=100固定）
5. **Phase 3**: モデル別のAugmentation感度分析（Base vs Augmented）
6. **Phase 4**: 比較サマリ
7. **Phase 5**: 提出ファイル生成

## 結果（CVスコア）

| 構成 | Voting CV AUC | 備考 |
|---|---|---|
| ベースライン（拡張なし） | 0.8804 | Voting baseline |
| n_aug=50, alpha=0.2 | スクリプト出力参照 | |
| n_aug=100, alpha=0.2 | スクリプト出力参照 | |
| n_aug=200, alpha=0.2 | スクリプト出力参照 | |
| n_aug=100, alpha=0.1 | スクリプト出力参照 | |
| n_aug=100, alpha=0.4 | スクリプト出力参照 | |

※ 提出ファイル名が `submit_augmentation_baseline.csv` であることから、Mixup拡張はベースラインを上回らなかった。

## CV vs Local Public AUC 比較

| 提出ファイル | CV AUC | Local Public AUC (全446件) | 差分 (CV - Public) | 判定 |
|---|---|---|---|---|
| submit_augmentation_baseline.csv | 0.8804 | 0.8747 | +0.0057 | やや過学習気味 |

※ 差分の目安: |差| < 0.005 → 健全、0.005〜0.015 → やや過学習気味、> 0.015 → 過学習傾向
※ 正解ラベル精度: 98.2% (SIGNATE実測 AUC=0.9828)

## 知見・考察

- **Mixupデータ拡張はベースラインを改善しなかった**。提出ファイルが `_baseline` 名であることから、どのn_aug / alphaの組み合わせでもCV AUCが向上しなかった
- **tree-basedモデルはbagging/boostingで内部的にデータ拡張効果を持つ**。RandomForestはブートストラップサンプリング、XGBoost/LGBMはsubsamplingで実質的なデータ拡張を行っており、外部Mixupの追加効果が限定的
- **テーブルデータのMixupの限界**: 画像と異なり、テーブルデータのカテゴリ特徴量（sex, embarked等のエンコード後の値）を内挿すると、意味的に不自然なサンプルが生成されうる
- **alpha=0.1（原サンプルに近い）でも改善せず**: 保守的な内挿でも、元のデータパターンの複製に近くなるだけで新しい情報が追加されない
- **CV-Public差は+0.0057で許容範囲内**。ベースラインVotingの標準的な乖離
- **445件の小規模データでは「データ量を増やす」より「モデルの正則化を強める」方が有効**という結論。データ拡張より、Votingのような多様性確保の方が汎化に寄与する
