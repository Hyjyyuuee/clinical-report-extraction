"""Apply a reviewed corrections file to draft labels and write the gold file.

Usage:
    python -m cre.data.apply_corrections --draft data/gold/test.draft.jsonl \
        --corrections data/gold/corrections/test.txt --out data/gold/test.jsonl
"""

import argparse
import json
from pathlib import Path

from ..schema import ReportExtraction


def parse_side(value: str) -> str | None:
    return None if value in ("-", "") else value


def apply(findings: list[dict], op: str, report: str) -> None:
    kind, body = op[0], op[1:]
    if kind == "+":
        finding, status, side, evidence = body.split("/", 3)
        if evidence.lower() not in report.lower():
            raise ValueError(f"evidence not in report: {evidence!r}")
        new = {"finding": finding, "status": status, "laterality": parse_side(side),
               "location": None, "evidence": evidence}
        if not any((f["finding"], f["status"], f["laterality"]) == (finding, status, new["laterality"])
                   for f in findings):
            findings.append(new)
        return
    target, _, changes = body.partition(">")
    finding, status = target.split("/")
    matches = [f for f in findings if f["finding"] == finding and f["status"] == status]
    if not matches:
        raise ValueError(f"no {finding}/{status} to {'remove' if kind == '-' else 'change'}")
    for match in matches:
        if kind == "-":
            findings.remove(match)
        else:
            for change in changes.split(","):
                key, value = change.split("=")
                match[key] = parse_side(value) if key == "laterality" else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    corrections = {}
    for line in args.corrections.open(encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if line:
            report_id, ops = line.split(":", 1)
            corrections.setdefault(report_id.strip(), []).extend(
                op.strip() for op in ops.split(";") if op.strip())

    rows = [json.loads(line) for line in args.draft.open(encoding="utf-8")]
    seen = set()
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            findings = row["label"]["findings"]
            for op in corrections.get(row["id"], []):
                try:
                    apply(findings, op, row["report"])
                except ValueError as error:
                    raise SystemExit(f"report {row['id']}, op {op!r}: {error}")
            row["label"] = ReportExtraction.model_validate({"findings": findings}).model_dump(mode="json")
            seen.add(row["id"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    unknown = set(corrections) - seen
    if unknown:
        raise SystemExit(f"corrections for unknown report ids: {sorted(unknown)}")
    print(f"applied {sum(map(len, corrections.values()))} edits to {len(corrections)} of {len(rows)} reports -> {args.out}")


if __name__ == "__main__":
    main()
