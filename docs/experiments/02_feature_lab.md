# 特徴量エンジニアリング実験室（Feature Lab）

## このノートブックでやること

さまざまな特徴量の組み合わせを試し、全モデルを一括で回して「どの特徴量が本当に効くか」を検証する。
特徴量の"実験台"だ。

## 特徴量エンジニアリングとは何か

ゼロつく1では、入力（画像のピクセル値）をそのままニューラルネットに渡した。
NNは自分で良い特徴を学習してくれるからだ。

しかし、ロジスティック回帰や決定木などの「浅いモデル」は、**人間が良い入力を用意してあげないと性能が出ない**。
これが特徴量エンジニアリングだ。

```
生データ: [pclass, sex, age, sibsp, parch, fare, embarked]
                ↓ 特徴量エンジニアリング
拡張データ: [上記 + family_size, is_alone, log_fare,
            fare_per_person, pclass_sex, age_bin, is_child, ...]
```

たとえば `family_size = sibsp + parch + 1` は単純な足し算だが、「家族が一緒かどうかが生存に影響する」というドメイン知識を数値に変換している。

## `make_exp_builder()` の使い方

```python
from src.exp_features import make_exp_builder, EXP_PRESETS

# フラグを切り替えて特徴量セットを構成
fb = make_exp_builder(
    missing_flags=True,       # 欠損フラグ
    age_bins="rule",          # 年齢ビニング
    fare_bins="quantile",     # 運賃ビニング
    interactions=True,        # 交互作用
    polynomial=False,         # 多項式
    group_stats=True,         # グループ統計量
    freq_encoding=False,      # 頻度エンコーディング
    rank_features=False,      # ランク特徴量
    domain_features=True,     # ドメイン特徴量
)

# プリセットも用意されている
fb = make_exp_builder(**EXP_PRESETS["recommended"])
```

### プリセット一覧

| プリセット | 特徴量数 | 方針 |
|---|---|---|
| `minimal` | 少なめ | 欠損フラグだけ追加 |
| `recommended` | 中程度 | 実用的な組み合わせ（推奨） |
| `kitchen_sink` | 最大 | 全部入り（過学習リスクあり） |

## 8カテゴリの特徴量カタログ

| # | カテゴリ | 特徴量例 | fold-aware | 説明 |
|---|---|---|---|---|
| 1 | 欠損フラグ | `age_missing`, `fare_missing` | - | 補完前に「欠損していたこと自体」を情報として保存 |
| 2 | ビニング | `age_bin`, `fare_qbin` | quantileのみ | 連続値をカテゴリに区切る（子供/大人/高齢者など） |
| 3 | 交互作用 | `age_pclass`, `fare_pclass`, `female_1` | - | 2つの特徴量の掛け合わせ |
| 4 | 多項式 | `age_sq`, `fare_sq`, `age_fare` | - | 非線形関係を捉える |
| 5 | グループ統計 | `fare_diff_pclass`, `age_z_sex` | **Yes** | 集団の中央値との差分やz-score |
| 6 | 頻度エンコーディング | `pclass_freq` | **Yes** | カテゴリの出現頻度を数値に変換 |
| 7 | ランク | `fare_pctile_in_pclass` | **Yes** | 客室クラス内での運賃の順位(0〜1) |
| 8 | ドメイン特徴量 | `is_child`, `is_mother`, `fare_zero` | - | タイタニック固有の知識を反映 |

### fold-aware（リーク防止）とは

「**Yes**」のカテゴリは、CVの各fold内で **学習データのみ** から統計量を計算する。
こうしないと、バリデーションデータの情報が学習時に漏れ込んでしまう（データリーク）。

```
[fold-aware の流れ]
fold 1: train(fold2-5)の中央値を計算 → train/valに適用
fold 2: train(fold1,3-5)の中央値を計算 → train/valに適用
  ...
→ 各foldで独立した統計量を使うので、リークなし
```

## ノートブックの流れ

1. **Setup**: ライブラリ・モデル定義の読み込み
2. **Data Loading**: 生データ読み込み
3. **Feature Builder**: `make_exp_builder()` でベースライン(v1)と実験用ビルダーを作成
4. **Model Definitions**: LogReg, RF, XGB, LGBM, Voting の5モデルを定義
5. **Cross-Validation**: ベースライン vs 実験で全モデルのCV AUCを一括比較
6. **Results Comparison**: 差分テーブルと棒グラフで可視化
7. **Submit**: ベースラインより改善していればベストモデルで提出ファイル生成

## 実験結果

この実験ではベースライン(v1)と差がなかった：

```
AUC improved in 0/5 models (mean diff: +0.0000)
→ 追加した特徴量がノイズになるか、元の特徴量と重複していた可能性
```

**これも重要な学び**だ。特徴量を増やせば必ず良くなるわけではない。

## ポイント・学び

- **特徴量エンジニアリングは仮説→検証のサイクル**。「効きそう」と思って作った特徴量も、CVで確認するまで本当に有効かはわからない
- **特徴量を増やしすぎると過学習する**。445件のデータに対して特徴量が多すぎると、モデルがノイズを学習してしまう
- **fold-aware は地味だが重要**。グループ統計量や頻度エンコーディングを正しく計算しないと、CVスコアが楽観的になり、本番で性能が下がる
- **「効果なし」も立派な実験結果**。何が効かないかを知ることで、次に試すべき方向が見えてくる
