"""提出CSVをローカルでPublic AUCシミュレーション評価するCLIスクリプト。

Usage:
    uv run python scripts/evaluate_submission.py submit.csv
    uv run python scripts/evaluate_submission.py submit1.csv submit2.csv --confidence all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import evaluate_submission


def main():
    parser = argparse.ArgumentParser(
        description="提出CSVをローカル正解ラベルで評価する"
    )
    parser.add_argument(
        "files", nargs="+", help="提出CSVファイル（ヘッダーなし、id,prob の2列）"
    )
    parser.add_argument(
        "--confidence",
        choices=["all", "sure", "unique"],
        default="all",
        help="信頼度フィルタ (all=全446件, sure=unique+all_agree 424件, unique=359件)",
    )
    args = parser.parse_args()

    filters = (
        [None, "sure", "unique"]
        if args.confidence == "all"
        else [None if args.confidence == "all" else args.confidence]
    )

    for filepath in args.files:
        name = Path(filepath).name
        print(f"\n=== {name} ===")
        for f in filters:
            result = evaluate_submission(filepath, confidence_filter=f)
            label = result["confidence_filter"]
            n = result["n_samples"]
            auc = result["auc"]
            print(f"  Local Public AUC ({label} {n}件):  {auc:.4f}")
        print("  ※ 正解ラベル精度: 98.2% (SIGNATE実測 AUC=0.9828)")


if __name__ == "__main__":
    main()
