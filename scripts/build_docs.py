"""README から docs/ を自動生成するスクリプト。

変換ルール:
  - experiments/best/README.md → docs/results.md（画像パスを assets/images/ に変換）
  - experiments/*/README.md   → docs/experiments/{dirname}.md
  - README.md                → docs/index.md（セットアップ省略 + ヒーローセクション追加）
  - experiments/best/*.png   → docs/assets/images/ にコピー
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
EXPERIMENTS = ROOT / "experiments"
BEST = EXPERIMENTS / "best"
IMAGES_DST = DOCS / "assets" / "images"


def ensure_dirs() -> None:
    """必要なディレクトリを作成。"""
    for d in [DOCS / "experiments", DOCS / "stylesheets", IMAGES_DST]:
        d.mkdir(parents=True, exist_ok=True)


def copy_images() -> None:
    """experiments/best/*.png → docs/assets/images/ にコピー。"""
    for png in sorted(BEST.glob("*.png")):
        dst = IMAGES_DST / png.name
        if not dst.exists() or png.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(png, dst)
            print(f"  copied: {png.name}")


def convert_best_readme() -> None:
    """experiments/best/README.md → docs/results.md（画像パス変換）。"""
    src = BEST / "README.md"
    if not src.exists():
        return
    text = src.read_text(encoding="utf-8")

    # 画像パス変換: ![alt](xxx.png) → ![alt](assets/images/xxx.png)
    text = re.sub(
        r"!\[([^\]]*)\]\(([^/)][^)]*\.png)\)",
        r"![\1](assets/images/\2)",
        text,
    )

    (DOCS / "results.md").write_text(text, encoding="utf-8")
    print("  generated: docs/results.md")


def fix_cross_experiment_links(text: str) -> str:
    """実験間の相対リンクを docs 構成に変換。

    ../01_preprocess/README.md → 01_preprocess.md
    ../20_ensemble/README.md   → 20_ensemble.md
    """
    text = re.sub(
        r"\.\./(\d{2}_[^/]+)/README\.md",
        r"\1.md",
        text,
    )
    return text


def convert_experiment_readmes() -> None:
    """experiments/*/README.md → docs/experiments/{dirname}.md。"""
    for readme in sorted(EXPERIMENTS.glob("*/README.md")):
        dirname = readme.parent.name
        if dirname == "best":
            continue
        dst = DOCS / "experiments" / f"{dirname}.md"
        text = readme.read_text(encoding="utf-8")
        text = fix_cross_experiment_links(text)
        dst.write_text(text, encoding="utf-8")
        print(f"  generated: docs/experiments/{dirname}.md")


def build_experiment_index() -> None:
    """docs/experiments/index.md — 実験一覧ページ。"""
    categories = {
        "データ分析・前処理": [
            ("00_eda", "EDA（探索的データ分析）"),
            ("01_preprocess", "前処理（v0 vs v1）"),
            ("02_feature_lab", "特徴量実験室"),
        ],
        "個別モデル": [
            ("10_logreg", "ロジスティック回帰"),
            ("11_rf", "ランダムフォレスト"),
            ("12_xgb", "XGBoost"),
            ("13_lgbm", "LightGBM"),
            ("14_nn", "ニューラルネットワーク（MLP）"),
        ],
        "アンサンブル": [
            ("20_ensemble", "Voting / Stacking"),
            ("21_tuning", "Optuna HPチューニング"),
            ("22_feature_review", "特徴量見直し・分布分析"),
        ],
        "高度アンサンブル": [
            ("23_hp_retune_domain_missing", "domain+missing HP再チューニング"),
            ("24_catboost", "CatBoost + 5モデルVoting"),
            ("25_advanced_stacking", "Advanced Stacking（Nested CV）"),
            ("26_multi_seed", "マルチシード平均化"),
            ("27_repeated_cv", "Repeated-CV ロバストチューニング"),
            ("28_rank_ensemble", "ランクアンサンブル"),
        ],
        "高度手法": [
            ("29_target_encoding", "ターゲットエンコーディング"),
            ("30_calibration", "確率キャリブレーション"),
            ("31_stability_selection", "安定特徴量選択"),
            ("32_pseudo_labeling", "合意ベース擬似ラベリング"),
            ("33_augmentation", "Mixup データ拡張"),
            ("34_bayesian", "ベイジアンモデル平均"),
        ],
        "最終最適化": [
            ("35_probability_blend", "確率ブレンド最適化"),
            ("36_hard_case_analysis", "Hard Case分析 + 外部データ特徴量"),
        ],
    }

    lines = [
        "# 実験一覧",
        "",
        "全36実験の記録。各実験の詳細は個別ページを参照。",
        "",
    ]

    for cat, exps in categories.items():
        lines.append(f"## {cat}")
        lines.append("")
        lines.append("| # | 実験 | 詳細 |")
        lines.append("|---|------|------|")
        for dirname, title in exps:
            num = dirname.split("_")[0]
            lines.append(f"| {num} | [{title}]({dirname}.md) | `experiments/{dirname}/` |")
        lines.append("")

    (DOCS / "experiments" / "index.md").write_text("\n".join(lines), encoding="utf-8")
    print("  generated: docs/experiments/index.md")


def convert_root_readme() -> None:
    """README.md → docs/index.md（セットアップ省略 + ヒーローセクション追加）。"""
    src = ROOT / "README.md"
    if not src.exists():
        return

    text = src.read_text(encoding="utf-8")

    # セットアップ〜ローカルPublicスコアシミュレーションのセクションを除去
    # "## セットアップ" から "## 実験の流れ" の直前まで
    text = re.sub(
        r"## セットアップ\n.*?(?=## 実験の流れ)",
        "",
        text,
        flags=re.DOTALL,
    )

    # "## ローカル Public スコアシミュレーション" セクションも除去
    text = re.sub(
        r"## ローカル Public スコアシミュレーション\n.*?(?=## )",
        "",
        text,
        flags=re.DOTALL,
    )

    # ヒーローセクションを冒頭に追加
    hero = """\
<div class="hero-banner" markdown>

## SIGNATE タイタニック生存予測

**Public AUC 0.8828** 達成 — 36実験の軌跡

[ベスト結果を見る :material-arrow-right:](results.md){ .md-button .md-button--primary }

</div>

!!! success "ベストスコア: Public AUC 0.8828"
    外部データ特徴量強化 + Cross-experiment Blend で達成。
    Local推定 AUC 0.8801 に対して +0.0027 の上振れ。

"""

    # 既存の "# SIGNATE タイタニック生存予測" タイトルを置換
    text = re.sub(
        r"^# SIGNATE タイタニック生存予測\n+",
        hero,
        text,
    )

    (DOCS / "index.md").write_text(text, encoding="utf-8")
    print("  generated: docs/index.md")


def main() -> None:
    print("Building docs...")
    ensure_dirs()
    copy_images()
    convert_best_readme()
    convert_experiment_readmes()
    build_experiment_index()
    convert_root_readme()
    print("Done!")


if __name__ == "__main__":
    main()
