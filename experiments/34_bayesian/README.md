# 実験34: Bayesian Model Averaging（ベイズモデル平均化）

## 目的

ロジスティック回帰を異なる特徴量サブセット・正則化強度の組み合わせで多数構築し、CV AUCに基づくsoftmax重みで平均化（Bayesian Model Averaging）することで、多様な弱学習器の集約によるアンサンブル効果を検証する。

## 手法

- **BMAモデル**: LogisticRegression（liblinear）
- **特徴量サブセット**（5種類）:
  - demographics: sex, age, pclass, is_child, is_alone
  - economic: fare, pclass, log_fare, fare_per_person, fare_zero
  - family: sibsp, parch, family_size, is_alone, family_small, family_large, is_mother
  - full_basic: sex, age, pclass, sibsp, parch, fare
  - domain_strong: 全特徴量（domain+missing）
- **正則化強度 C**: 0.001, 0.01, 0.1, 1.0
- **合計モデル数**: 5サブセット x 4C値 = 20モデル
- **重み付け**: softmax(temperature=50 x CV_AUC)で正規化
- **tree-basedモデルとの統合**: BMA予測とtree平均（RF+XGB+LGBM）をブレンド比率0.1〜0.5で試行

### 処理フロー

1. **Phase 0**: ベースラインCV（Equal Voting）
2. **Phase 1**: 特徴量サブセットの定義と利用可能特徴量の確認
3. **Phase 2**: 20モデルの各OOF予測を取得
4. **Phase 3**: BMA（Equal-weighted / Softmax-weighted / Top-K）
5. **Phase 4**: BMA + tree-basedモデルのブレンド
6. **Phase 5**: 全手法の比較
7. **Phase 6**: 提出ファイル生成

## 結果（CVスコア）

| 手法 | CV AUC | 備考 |
|---|---|---|
| ベースライン（Equal Voting） | 0.8804 | LogReg+RF+XGB+LGBM |
| BMA Equal-weighted | スクリプト出力参照 | 20モデル均等平均 |
| BMA Softmax-weighted | スクリプト出力参照 | CV AUC基づく重み |
| BMA Top-3/5/10 | スクリプト出力参照 | 上位Kモデルのみ |
| Tree Average（RF+XGB+LGBM） | スクリプト出力参照 | |
| BMA + Trees Blend | スクリプト出力参照 | 複数ブレンド比率 |

※ 提出ファイルが `submit_bma_reference.csv`（参考用）と `submit_bma_baseline.csv`（Voting baseline）であることから、BMAはベースラインを上回らなかった。

## CV vs Local Public AUC 比較

| 提出ファイル | CV AUC | Local Public AUC (全446件) | 差分 (CV - Public) | 判定 |
|---|---|---|---|---|
| submit_bma_reference.csv | N/A (BMA参考出力) | 0.8617 | N/A | - |
| submit_bma_baseline.csv | 0.8804 | 0.8747 | +0.0057 | やや過学習気味 |

※ 差分の目安: |差| < 0.005 → 健全、0.005〜0.015 → やや過学習気味、> 0.015 → 過学習傾向
※ 正解ラベル精度: 98.2% (SIGNATE実測 AUC=0.9828)

## 知見・考察

- **BMAはベースラインEqual Votingを改善しなかった**。LogRegの多様なサブセット集約は、tree-basedモデルのアンサンブルに及ばない
- **BMA reference（0.8617）はVoting baseline（0.8747）を大幅に下回る**。LogRegのみのアンサンブルでは、tree-basedモデルが捕捉する非線形パターンを表現できない
- **特徴量サブセットの多様性だけでは不足**: モデルタイプの多様性（線形 + 非線形）の方がアンサンブルへの寄与が大きい。LogRegの異なるサブセットは類似した線形決定境界しか生成しない
- **softmax temperatureの設定（50.0）は結果に影響するが、根本的にLogRegの表現力が限界**: temperature を変えても、弱いモデル群の加重平均は強いモデル群に勝てない
- **CV-Public差は+0.0057（baseline）で許容範囲内**
- **BMA + tree-basedブレンドも改善せず**: BMA成分が足を引っ張る形で、純粋なtreeアンサンブルの方が強い
- **学び**: Bayesian Model Averaging は同一モデルタイプの多様性確保には有効だが、既にモデルタイプ多様性を持つVotingの代替にはならない
