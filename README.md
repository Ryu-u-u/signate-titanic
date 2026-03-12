# SIGNATE タイタニック生存予測

SIGNATEタイタニックコンペ用プロジェクト。

## 構成

- `data/raw/` — 元データ（不変）
- `data/processed/` — 前処理済みデータ
- `experiments/` — 実験フォルダ（ノートブック・出力を同居）
- `src/` — 共通ライブラリ
- `configs/` — ハイパーパラメータ設定

## セットアップ

```bash
pip install -r requirements.txt
```

## 使い方

```python
from src.data import load_train
from src.features import preprocess
from src.utils import seed_everything

seed_everything()
df = load_train()
df = preprocess(df)
```
