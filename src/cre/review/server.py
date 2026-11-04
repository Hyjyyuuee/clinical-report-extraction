"""Local annotation tool for correcting draft labels into gold labels.

Reads data/gold/{split}.draft.jsonl and writes reviewed rows to data/gold/{split}.jsonl,
which is what `cre.evaluate --gold` consumes. Only reviewed reports go into the gold file,
so you can stop at any point and evaluate on what has been checked so far.

Usage:
    python -m cre.review.server            # then open http://127.0.0.1:8765
"""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

from ..schema import FindingType, Laterality, ReportExtraction, Status

PAGE = Path(__file__).with_name("index.html")
SPLITS = ("test", "dev")
_lock = threading.Lock()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


class Store:
    def __init__(self, gold_dir: Path):
        self.gold_dir = gold_dir

    def items(self, split: str) -> list[dict]:
        reviewed = {row["id"]: row for row in read_jsonl(self.gold_dir / f"{split}.jsonl")}
        items = []
        for row in read_jsonl(self.gold_dir / f"{split}.draft.jsonl"):
            done = reviewed.get(row["id"])
            items.append({
                "id": row["id"],
                "report": row["report"],
                "label": (done or row)["label"],
                "reviewed": done is not None,
            })
        return items

    def save(self, split: str, report_id: str, label: dict) -> None:
        draft = {row["id"]: row for row in read_jsonl(self.gold_dir / f"{split}.draft.jsonl")}
        if report_id not in draft:
            raise KeyError(report_id)
        clean = ReportExtraction.model_validate(label).model_dump(mode="json")
        gold_path = self.gold_dir / f"{split}.jsonl"
        with _lock:
            reviewed = {row["id"]: row for row in read_jsonl(gold_path)}
            reviewed[report_id] = {**draft[report_id], "label": clean}
            # Keep draft order so diffs of the gold file stay readable.
            ordered = [reviewed[i] for i in draft if i in reviewed]
            tmp = gold_path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for row in ordered:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            tmp.replace(gold_path)


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload) -> None:
            self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _split(self) -> str | None:
            split = parse_qs(urlparse(self.path).query).get("split", ["test"])[0]
            return split if split in SPLITS else None

        def do_GET(self) -> None:
            route = urlparse(self.path).path
            if route == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif route == "/api/schema":
                self._json(200, {
                    "finding": [f.value for f in FindingType],
                    "status": [s.value for s in Status],
                    "laterality": [l.value for l in Laterality],
                    "splits": list(SPLITS),
                })
            elif route == "/api/items" and (split := self._split()):
                self._json(200, store.items(split))
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/save" or not (split := self._split()):
                return self._json(404, {"error": "not found"})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            try:
                store.save(split, body["id"], body["label"])
            except ValidationError as error:
                return self._json(400, {"error": error.errors(include_url=False)})
            except KeyError:
                return self._json(404, {"error": "unknown report id"})
            self._json(200, {"ok": True})

        def log_message(self, *args) -> None:
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-dir", type=Path, default=Path("data/gold"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(Store(args.gold_dir)))
    print(f"Review tool running at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
