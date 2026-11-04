"""Join API predictions with their reports to form silver labels.

Usage:
    python -m cre.predict --split data/processed/train.jsonl --backend api --model deepseek-chat \
        --few-shot --out outputs/predictions/train-silver.jsonl
    python -m cre.data.silver --split data/processed/train.jsonl \
        --pred outputs/predictions/train-silver.jsonl --out data/silver/train.jsonl
"""

import argparse
import json
from pathlib import Path

from ..evaluate import parse_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    preds = {row["id"]: row["raw"] for row in map(json.loads, args.pred.open(encoding="utf-8"))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    with args.out.open("w", encoding="utf-8") as f:
        for row in map(json.loads, args.split.open(encoding="utf-8")):
            parsed = parse_output(preds.get(row["id"], ""))
            if parsed is None:
                dropped += 1
                continue
            row["label"] = parsed.model_dump(mode="json")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    print(f"kept {kept}, dropped {dropped} unparseable -> {args.out}")


if __name__ == "__main__":
    main()
