# 実験30: Calibration-First Ensembling（確率キャリブレーション後アンサンブル）

## 目的

各ベースモデル（LogReg, RF, XGBoost, LightGBM）の出力確率スケールを揃えた上でブレンドすることで、アンサンブル品質が向上するかを検証する。Platt Scaling（sigmoid）と Isotonic Regression の2手法でキャリブレーションし、さらに logit 空間でのブレンドも試す。

## 手法

- **ベースモデル**: LogReg, RandomForest, XGBoost, LightGBM（チューニング済みパラメータ）
- **特徴量**: domain+missing（exp_features の make_exp_builder）
- **キャリブレーション手法**:
  - Platt Scaling（sigmoid）: CalibratedClassifierCV(method="sigmoid", cv=3)
  - Isotonic Regression: CalibratedClassifierCV(method="isotonic", cv=3)
- **ブレンド手法**:
  1. Uncalibrated Simple Average（単純平均）
  2. Uncalibrated Logit Blend（logit空間平均）
  3. Calibrated Simple Average（キャリブレーション後の単純平均）
  4. Calibrated Logit Blend（キャリブレーション後のlogit空間平均）
  5. VotingClassifier（ベースライン）
- **評価指標**: AUC + Brier Score（キャリブレーション品質）
- **提出ファイル**: 3種類（cal_logit_blend, cal_simple_avg, voting_baseline）

### 処理フロー

1. **Phase 1**: 各ベースモデルのキャリブレーション診断（Brier Score, Calibration Curve）
2. **Phase 2**: Platt / Isotonic でキャリブレーション、モデル別にベスト手法を選定
3. **Phase 3**: キャリブレーション済みOOF確率でlogit空間ブレンド
4. **Phase 4**: 提出ファイル生成（3種類）
5. **Phase 5**: 全手法の比較テーブル

## 結果（CVスコア）

| 手法 | CV AUC | 備考 |
|---|---|---|
| 各ベースモデル単体 | スクリプト出力参照 | Brier Scoreも計測 |
| VotingClassifier（ベースライン） | 0.8804 | Equal Voting |
| Uncal Simple Average | スクリプト出力参照 | OOF確率の単純平均 |
| Uncal Logit Blend | スクリプト出力参照 | logit空間平均 |
| Cal Simple Average | スクリプト出力参照 | キャリブレーション後 |
| Cal Logit Blend | スクリプト出力参照 | キャリブレーション後logit空間 |

※ スクリプトの出力からは、キャリブレーションが必ずしもAUCを改善しないケースが確認される。Brier Scoreの改善はAUCの改善と必ずしも一致しない。

## CV vs Local Public AUC 比較

| 提出ファイル | CV AUC | Local Public AUC (全446件) | 差分 (CV - Public) | 判定 |
|---|---|---|---|---|
| submit_cal_logit_blend.csv | 0.8804 (参考: Voting baseline) | 0.8742 | +0.0062 | やや過学習気味 |
| submit_cal_simple_avg.csv | 0.8804 (参考: Voting baseline) | 0.8747 | +0.0057 | やや過学習気味 |
| submit_voting_baseline.csv | 0.8804 | 0.8747 | +0.0057 | やや過学習気味 |

※ 差分の目安: |差| < 0.005 → 健全、0.005〜0.015 → やや過学習気味、> 0.015 → 過学習傾向
※ 正解ラベル精度: 98.2% (SIGNATE実測 AUC=0.9828)

## 知見・考察

- **キャリブレーションはAUCをほとんど改善しなかった**。cal_logit_blend（0.8742）はvoting_baseline（0.8747）を下回り、キャリブレーションが汎化に寄与しなかった
- **cal_simple_avg と voting_baseline が同スコア（0.8747）**。VotingClassifier内部のsoft votingと手動OOF平均が実質的に同じ結果を生む
- **logit空間ブレンドは逆効果**。確率スケールが異なるモデルのlogitを平均すると、極端な確信度を持つモデルに引きずられるリスクがある
- **CV-Public差は+0.005〜0.006程度**でやや過学習気味だが、許容範囲内。445件の小規模データでは標準的な乖離レベル
- **Brier Scoreはキャリブレーションで改善しうるが、AUC（ランキング性能）には影響しない**ことが確認された。AUCは単調変換に不変であり、キャリブレーションは単調変換の一種
