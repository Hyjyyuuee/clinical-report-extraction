"""Parse the IU X-Ray (Open-i) report XML files into a JSONL corpus with splits.

Usage:
    python -m cre.data.prepare_iu --raw data/raw/ecgen-radiology --out data/processed
"""

import argparse
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from ..prompts import format_report


def parse_report(path: Path) -> dict | None:
    root = ET.parse(path).getroot()
    sections = {
        (node.get("Label") or "").upper(): (node.text or "").strip()
        for node in root.iter("AbstractText")
    }
    findings, impression = sections.get("FINDINGS", ""), sections.get("IMPRESSION", "")
    if not findings and not impression:
        return None
    # MeSH terms are weak labels assigned by the dataset curators; useful for sanity checks.
    mesh = [node.text for node in root.iter("major") if node.text]
    return {
        "id": path.stem,
        "findings": findings,
        "impression": impression,
        "report": format_report(findings, impression),
        "mesh": mesh,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--dev-size", type=int, default=200)
    args = parser.parse_args()

    records = [r for p in sorted(args.raw.glob("*.xml")) if (r := parse_report(p))]
    # Many IU reports are verbatim duplicates ("No acute cardiopulmonary abnormality.");
    # keep one per text so test cases are not leaked into training.
    unique = list({r["report"]: r for r in records}.values())
    random.Random(args.seed).shuffle(unique)

    splits = {
        "test": unique[: args.test_size],
        "dev": unique[args.test_size : args.test_size + args.dev_size],
        "train": unique[args.test_size + args.dev_size :],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"parsed {len(records)} reports, {len(unique)} unique: "
          + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))


if __name__ == "__main__":
    main()
