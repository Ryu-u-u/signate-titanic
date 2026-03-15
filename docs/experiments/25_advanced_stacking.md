# 実験25: Advanced Stacking (Nested CV)

## 目的

Equal Voting の等重みに代えて、Stacking（メタ学習）で最適なブレンディング重みを学習し、予測精度を向上させる。
sklearn StackingClassifier と手動 Nested CV の2手法を比較し、Equal Voting を上回るかを検証する。

## 手法

### 特徴量
- domain+missing 特徴量セット（約23特徴量）
  - ドメイン特徴量: is_child, is_mother 等
  - 欠損フラグ: age_missing, cabin_missing 等

### ベースモデル (Level-0)
- LogisticRegression (StandardScaler + Pipeline)
- RandomForest
- XGBoost (GPU: device="cuda")
- LightGBM

### メタ学習器 (Level-1)
- LogisticRegression (C をグリッドサーチ: 0.01, 0.1, 1.0, 10.0)

### フェーズ構成

**ベースライン: Equal Voting**
- 4モデル soft voting（等重み）の CV AUC を計測

**Phase 1: sklearn StackingClassifier**
- LogReg メタ学習器 (C=0.1)、内部5-fold CV でメタ特徴量生成
- passthrough=False（メタ特徴量のみ使用）

**Phase 2: Meta-Learner Configuration Search**
- C値: [0.01, 0.1, 1.0, 10.0]
- passthrough: [False, True]
- 計8構成をグリッドサーチし、最良構成を特定

**Phase 3: Manual Nested CV**
- 外側5-fold × 内側5-fold の完全な Nested CV
- 各外側 fold で:
  1. 内側 OOF で4モデルのメタ特徴量を生成
  2. メタ学習器を学習・外側テストを予測
  3. メタ学習器の重み（coef_）を出力
- 複数の C 値で同時評価

**Phase 4: Stacking vs Equal Voting 比較**
- Equal Voting / sklearn Stacking (ベスト構成) / Manual Nested CV (ベスト C) を比較

**Phase 5: 提出ファイル生成**
1. `submit_stacking.csv`: 手動 Stacking（全 OOF → メタ学習 → テスト予測）
2. `submit_voting_baseline.csv`: Equal Voting ベースライン
3. `submit_sklearn_stacking.csv`: sklearn StackingClassifier (ベスト構成)

## 結果（CVスコア）

CV AUC はスクリプト実行時に出力される。以下の情報が出力される:

```
ベースライン: Equal Voting AUC + 各単体モデル AUC
Phase 1: sklearn Stacking (C=0.1) AUC + vs Voting 差分
Phase 2: 8構成の AUC + vs Voting 差分一覧
Phase 3: Manual Nested CV の per-fold AUC + Mean AUC + Overall OOF AUC
         + C値ごとの比較表 + メタ学習器の重み
Phase 4: 全手法の比較表
```

ベースラインの4モデル Voting CV AUC は実験22で 0.8804 が報告されている。

## CV vs Local Public AUC 比較

| 提出ファイル | CV AUC | Local Public AUC (全446件) | 差分 (CV - Public) | 判定 |
|---|---|---|---|---|
| submit_stacking.csv | N/A | 0.8748 | N/A | - |
| submit_voting_baseline.csv | N/A | 0.8747 | N/A | - |
| submit_sklearn_stacking.csv | N/A | 0.8745 | N/A | - |

※ CV AUC はスクリプト実行時に出力されるが、ログファイルとして保存されていないため N/A。
※ 差分の目安: |差| < 0.005 → 健全、0.005〜0.015 → やや過学習気味、> 0.015 → 過学習傾向
※ 正解ラベル精度: 98.2% (SIGNATE実測 AUC=0.9828)

## 知見・考察

- **Stacking は Equal Voting を上回らなかった**。3提出ファイル全てで Local Public AUC が Voting ベースライン (0.8747) と同等以下
- submit_stacking.csv (0.8748) が最高だが、submit_voting_baseline.csv (0.8747) との差はわずか +0.0001 であり、実質的な改善とは言えない
- **Stacking のメタ学習器が過学習した可能性がある**。445件の小規模データでは、OOF メタ特徴量（4次元）に対してメタ学習器が学習するデータが不十分
- sklearn StackingClassifier (0.8745) < Manual Nested CV (0.8748) であり、手動実装の方がやや優れているが、差は誤差範囲内
- passthrough オプション（元特徴量をメタ特徴量に追加）が過学習を悪化させる可能性がある。少データではメタ学習器への入力次元を最小限に抑えるべき
- **Equal Voting の単純さが445件の小規模データに適している**。重み学習による改善余地は限られ、等重みの方が汎化性能が安定する
- このデータ規模では「より賢い」アンサンブル手法よりも、特徴量エンジニアリングやモデル多様性の確保が精度向上の主要ドライバーである
