"""Turn labelled reports into chat-formatted SFT data.

Input rows need "report" and "label" (a ReportExtraction dict). Labels come from
GPT/DeepSeek pre-annotation (silver) for train, and human-corrected labels (gold) for dev/test.

Usage:
    python -m cre.data.build_sft --labels data/silver/train.jsonl --out data/sft/train.jsonl
"""

import argparse
import json
from pathlib import Path

from ..prompts import build_messages
from ..schema import ReportExtraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    with args.out.open("w", encoding="utf-8") as f:
        for row in map(json.loads, args.labels.open(encoding="utf-8")):
            try:
                label = ReportExtraction.model_validate(row["label"])
            except Exception:
                skipped += 1
                continue
            # Prompt/completion format so TRL computes loss on the JSON answer only.
            example = {
                "prompt": build_messages(row["report"]),
                "completion": [{"role": "assistant", "content": label.model_dump_json()}],
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            kept += 1
    print(f"kept {kept}, skipped {skipped} invalid labels -> {args.out}")


if __name__ == "__main__":
    main()
