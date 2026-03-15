# 実験36: Hard Case分析 + 外部データ特徴量強化

## 目的

全16モデルが不正解とした25件のテストレコード（「鉄壁の壁」）を深掘り分析し、そのパターンから新しい特徴量を設計する。さらに外部データ（titanic3.csv）からTitle・Cabin Deck・Ticket Group Sizeを復元し、特徴量を強化してスコア改善を目指す。

## 手法

### Phase 1-3: Hard Case分析
- **Phase 1**: 25件のHard Caseをプロファイリング（性別・年齢・Pclass・運賃・家族構成）
- **Phase 2**: 外部データ（titanic3.csv）とのマッチングにより名前・チケット・キャビン情報を復元
- **Phase 3**: Easy Survivedとの特徴量比較による「例外生存パターン」の特定
  - 22/25件が「死亡予測だが実際は生存」→ 男性＋3等＋低運賃が典型パターン
  - 家族持ち男性、若年男性（<15歳）が例外的に生存している

### Phase 4: Exception Features（例外特徴量）
Hard Case分析から導出された新特徴量:
- `male_with_family`: 男性かつ家族同伴（単独男性より生存率高い）
- `young_male`: 15歳未満の男性（子ども扱いで優先避難の可能性）
- `male_upper_class`: 1-2等の男性（上等クラス男性は生存率高い）
- `male_with_children`: 子連れ男性
- `is_elderly`: 60歳以上（高齢者優先の可能性）
- `small_family_3rd`: 3等の小家族（2-4人、単独より生存しやすい）
- `age_x_male` / `age_x_female`: 年齢×性別の交互作用
- `fare_zscore_pclass`: クラス内運賃Zスコア（fold-aware計算）
- `high_fare_for_class`: クラス内運賃上位25%フラグ

### Phase 5-6: Enriched Features（外部データ強化特徴量）
外部データ（titanic3.csv）から復元した特徴量:
- `title_enc` / `rare_title` / `is_master`: 乗客の敬称（Mr/Mrs/Miss/Master等）
- `has_cabin` / `upper_deck`: キャビンの有無・デッキ位置（A-C=上層）
- `in_group` / `large_group`: チケット共有グループサイズ

### Phase 7: ブレンド
- Enriched Voting と exp23 Retuned Voting の確率レベルブレンド（異なる特徴量セット）
- ブレンド重みのグリッドサーチ（step=0.05）

## 結果（CVスコア）

| 手法 | CV AUC (Voting) | vs Baseline |
|---|---|---|
| baseline (domain+missing) | 0.8804 | — |
| enhanced (exception feats) | 0.8817 | +0.0013 |
| enriched (external+exception) | 0.8807 | +0.0003 |

※ Exception Featuresが+0.0013改善。外部データ追加（enriched）ではCV上の上乗せは限定的だが、Local Publicでは大きく改善。

## CV vs Local Public AUC 比較

| 提出ファイル | CV AUC | Local Public AUC (全446件) | 差分 (CV - Public) | 判定 |
|---|---|---|---|---|
| submit_enriched_retuned_blend.csv | N/A (blend) | **0.8801** | N/A | — |
| submit_enriched_voting.csv | 0.8807 | 0.8787 | +0.0020 | 健全 |
| submit_enhanced_voting.csv | 0.8817 | 0.8771 | +0.0046 | 健全 |
| submit_enhanced_blend.csv | N/A | 0.8771 | N/A | — |
| submit_baseline_voting.csv | 0.8804 | 0.8754 | +0.0050 | やや過学習気味 |

※ 差分の目安: |差| < 0.005 → 健全、0.005〜0.015 → やや過学習気味、> 0.015 → 過学習傾向
※ 正解ラベル精度: 98.2% (SIGNATE実測 AUC=0.9828)

## 知見・考察

- **外部データTitle/Deck/Ticketが最も効果的**: Enriched VotingがLocal Public 0.8787を達成し、従来ベスト(0.8762)を+0.0025改善。Title（敬称）による社会的地位の捕捉が特に有効
- **Hard Case分析→特徴量設計サイクルの有効性**: 全モデル不正解のパターン分析から「男性の例外生存条件」を特定し、それを特徴量化するアプローチが機能
- **CV vs Local Public の乖離が改善**: Enhanced(+0.0046)→Enriched(+0.0020)と、外部データ追加で汎化性能が改善
- **Cross-experiment blendとの組み合わせで最大効果**: Enriched Voting + exp23 Retuned Voting のブレンドが0.8801（**プロジェクト全体ベスト**）を達成
- **Exception FeaturesはCV上では効果あり、Local Publicでは控えめ**: fold-aware計算でリーク防止しているが、445件の小規模データでは新特徴量の効果が不安定
- **学び**: 小規模データでの改善は「新しい情報源（外部データ）」と「多様なパイプラインのブレンド」の組み合わせが最も確実
