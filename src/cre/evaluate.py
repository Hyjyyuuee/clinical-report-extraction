"""Score predictions against gold labels.

A finding matches when (finding, status, laterality) agree. Scores are reported
overall and per status, because negation and uncertainty are where models fail.

Two parsing modes are reported:
- strict (top-level keys): the whole output must validate against ReportExtraction;
  otherwise the report contributes no predictions.
- lenient (under "lenient"): the output only has to be a JSON object with a "findings"
  list; each finding that fails validation is dropped on its own and the rest are scored.

Usage:
    python -m cre.evaluate --gold data/gold/test.jsonl --pred outputs/predictions/deepseek-chat.jsonl
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from .schema import Finding, FindingType, ReportExtraction, Status


def parse_output(raw: str) -> ReportExtraction | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return ReportExtraction.model_validate_json(match.group(0))
    except ValidationError:
        return None


def parse_output_lenient(raw: str) -> tuple[ReportExtraction | None, int]:
    """Keep the valid findings of a JSON output; return (extraction, number of findings dropped)."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None, 0
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, 0
    items = data.get("findings") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return None, 0
    findings, dropped = [], 0
    for item in items:
        try:
            findings.append(Finding.model_validate(item))
        except ValidationError:
            dropped += 1
    return ReportExtraction(findings=findings), dropped


def keys(extraction: ReportExtraction) -> set[tuple]:
    # "other" is a catch-all with no fixed identity, so it is not scored.
    return {
        (f.finding.value, f.status.value, f.laterality.value if f.laterality else None)
        for f in extraction.findings
        if f.finding != FindingType.other
    }


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "support": tp + fn}


def score(gold: dict[str, set], parsed: dict[str, ReportExtraction | None]) -> dict:
    counts = Counter()
    for report_id, gold_keys in gold.items():
        extraction = parsed.get(report_id)
        pred_keys = keys(extraction) if extraction else set()
        for status in [None, *(s.value for s in Status)]:
            g = {k for k in gold_keys if status is None or k[1] == status}
            p = {k for k in pred_keys if status is None or k[1] == status}
            counts[(status, "tp")] += len(g & p)
            counts[(status, "fp")] += len(p - g)
            counts[(status, "fn")] += len(g - p)
    return {
        "overall": prf(counts[(None, "tp")], counts[(None, "fp")], counts[(None, "fn")]),
        **{s.value: prf(counts[(s.value, "tp")], counts[(s.value, "fp")], counts[(s.value, "fn")])
           for s in Status},
    }


def evaluate(gold_path: Path, pred_path: Path) -> dict:
    gold = {
        row["id"]: keys(ReportExtraction.model_validate(row["label"]))
        for row in map(json.loads, gold_path.open(encoding="utf-8"))
    }
    preds = {row["id"]: row["raw"] for row in map(json.loads, pred_path.open(encoding="utf-8"))}
    n = max(len(gold), 1)

    strict = {report_id: parse_output(preds.get(report_id, "")) for report_id in gold}
    lenient, dropped = {}, 0
    for report_id in gold:
        lenient[report_id], n_dropped = parse_output_lenient(preds.get(report_id, ""))
        dropped += n_dropped

    return {
        "n_reports": len(gold),
        "valid_json_rate": round(sum(v is not None for v in strict.values()) / n, 4),
        **score(gold, strict),
        "lenient": {
            "parseable_rate": round(sum(v is not None for v in lenient.values()) / n, 4),
            "dropped_findings": dropped,
            **score(gold, lenient),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.gold, args.pred), indent=2))


if __name__ == "__main__":
    main()
