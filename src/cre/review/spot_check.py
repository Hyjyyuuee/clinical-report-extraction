"""Build a self-contained HTML sheet for spot-checking a sample of label corrections.

Each item shows the report, the phrase the correction relies on (highlighted), what was
changed, and a plain-language question that can be answered by reading the English text.

Usage:
    python -m cre.review.spot_check --split test --n 16 --out outputs/spot_check.html
"""

import argparse
import html
import json
import random
import re
from pathlib import Path

ZH = {
    "consolidation": "实变", "infiltrate": "浸润", "opacity": "阴影", "pleural_effusion": "胸腔积液",
    "atelectasis": "肺不张", "pneumothorax": "气胸", "edema": "肺水肿", "cardiomegaly": "心脏增大",
    "nodule": "结节", "granuloma": "肉芽肿", "emphysema": "肺气肿", "pneumonia": "肺炎", "other": "其他",
    "present": "有", "absent": "没有", "uncertain": "不确定",
}

# (pattern in the English reason, category, what to check, words to highlight)
CATEGORIES = [
    ("granuloma", "肉芽肿", "原文里有没有 granuloma 或 granulomatous 这个词？有 → 同意。",
     r"granulom\w*"),
    ("maps to opacity", "airspace → 阴影", "原文写的是 airspace disease / airspace opacity，而不是 consolidation？是 → 同意。",
     r"air ?space \w+"),
    ("not emphysema", "不算肺气肿", "原文只写了 hyperexpanded / hyperinflated（肺膨胀），没有出现 emphysema 或 COPD？是 → 同意。",
     r"hyper\w+|emphysema\w*|COPD"),
    ("not atelectasis", "不算肺不张", "原文说的是 low volumes / volume loss / crowding，没有出现 atelectasis？是 → 同意。",
     r"low (lung )?volumes?|volume loss|crowding|atelectasis|hypoinflat\w*"),
    ("upper limits", "正常上限", "原文说心脏大小在 upper / top limits of normal（正常范围上限）？是 → 同意。",
     r"(upper|top) (limits? )?(of )?normal|top normal"),
    ("stable", "只说稳定", "原文只说 stable（稳定），没说 normal（正常）？是 → 同意。",
     r"stable"),
    ("hedged|or atelectasis|and/or|versus|suggest|Appearance", "不确定", "原文有 or / versus / may / suggest / XXXX representing 这类含糊的词？有 → 同意。",
     r"\b(or|versus|may|suggest\w*|possibl\w*|probabl\w*|XXXX represent\w*)\b"),
    ("missed", "漏标补上", "高亮的那句话，是不是确实在说这个发现（有或没有）？是 → 同意。", None),
]


def describe(op: str) -> str:
    kind, body = op[0], op[1:]
    if kind == "+":
        finding, status, side, _ = body.split("/", 3)
        side = "" if side == "-" else f"（{side}）"
        return f"新增：{ZH[finding]} = {ZH[status]}{side}"
    target, _, changes = body.partition(">")
    finding, status = target.split("/")
    if kind == "-":
        return f"删除：{ZH[finding]} = {ZH[status]}"
    parts = []
    for change in changes.split(","):
        key, value = change.split("=")
        parts.append({"finding": "类型", "status": "状态", "laterality": "左右"}[key]
                     + " → " + ZH.get(value, "未注明" if value == "-" else value))
    return f"修改：{ZH[finding]} = {ZH[status]}，" + "，".join(parts)


def load(split: str):
    reports = {json.loads(l)["id"]: json.loads(l)["report"]
               for l in open(f"data/gold/{split}.draft.jsonl", encoding="utf-8")}
    items = []
    for line in open(f"data/gold/corrections/{split}.txt", encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        body, _, reason = line.partition("#")
        rid, ops = body.split(":", 1)
        ops = [o.strip() for o in ops.split(";") if o.strip()]
        reason = reason.strip() or "missed"
        for pattern, name, question, highlight in CATEGORIES:
            if re.search(pattern, reason):
                items.append({"id": rid.strip(), "ops": ops, "reason": reason, "category": name,
                              "question": question, "highlight": highlight, "report": reports[rid.strip()]})
                break
    return items


def render_report(item: dict) -> str:
    text = html.escape(item["report"])
    spans = [op.split("/", 3)[3] for op in item["ops"] if op.startswith("+")]
    patterns = [re.escape(html.escape(s)) for s in spans]
    if item["highlight"]:
        patterns.append(item["highlight"])
    if patterns:
        text = re.sub("(" + "|".join(patterns) + ")", r"<mark>\1</mark>", text, flags=re.I)
    return text.replace("\n", "<br>")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("outputs/spot_check.html"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    items = load(args.split)
    by_category = {}
    for item in items:
        by_category.setdefault(item["category"], []).append(item)
    # Round-robin over categories so every kind of correction is represented.
    sample, pools = [], [rng.sample(v, len(v)) for v in by_category.values()]
    while len(sample) < args.n and any(pools):
        for pool in pools:
            if pool and len(sample) < args.n:
                sample.append(pool.pop())

    cards = "\n".join(f"""
<section class="card" data-id="{html.escape(it['id'])}">
  <div class="head"><span>第 {i} 题 · 报告 #{html.escape(it['id'])}</span><span class="tag">{it['category']}</span></div>
  <div class="report">{render_report(it)}</div>
  <div class="change">{'<br>'.join(html.escape(describe(op)) for op in it['ops'])}</div>
  <div class="q">✔ 检查：{it['question']}</div>
  <div class="vote">
    <label><input type="radio" name="v{i}" value="同意"> 同意</label>
    <label><input type="radio" name="v{i}" value="不同意"> 不同意</label>
    <label><input type="radio" name="v{i}" value="看不懂"> 看不懂</label>
  </div>
</section>""" for i, it in enumerate(sample, 1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(TEMPLATE.replace("{{CARDS}}", cards).replace("{{N}}", str(len(sample))),
                        encoding="utf-8")
    print(f"wrote {len(sample)} items -> {args.out}")


TEMPLATE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label Spot Check</title>
<style>
  :root { --bg:#f7f7f5; --panel:#fff; --text:#1f2328; --muted:#6a737d; --border:#d8dadd; --mark:#fff3a3; --accent:#2f6fdb; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#16181c; --panel:#1f2227; --text:#e6e8eb; --muted:#9aa3ad; --border:#343a42; --mark:#5c4d0a; --accent:#6ea1ff; }
  }
  * { box-sizing: border-box; }
  body { margin:0; padding:24px 16px 80px; background:var(--bg); color:var(--text); font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif; }
  main { max-width:820px; margin:0 auto; }
  h1 { font-size:20px; margin:0 0 8px; }
  .intro, .card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px 18px; margin-bottom:16px; }
  .intro ol { margin:6px 0 0; padding-left:20px; }
  .head { display:flex; justify-content:space-between; color:var(--muted); font-size:13px; margin-bottom:8px; }
  .tag { border:1px solid var(--border); border-radius:10px; padding:0 8px; }
  .report { font-family:ui-monospace,Consolas,monospace; font-size:13.5px; background:var(--bg); padding:10px 12px; border-radius:6px; }
  mark { background:var(--mark); color:inherit; border-radius:3px; padding:0 2px; }
  .change { margin:10px 0 4px; font-weight:600; }
  .q { color:var(--muted); }
  .vote { display:flex; gap:18px; margin-top:8px; flex-wrap:wrap; }
  button { font:inherit; padding:8px 16px; border-radius:6px; border:1px solid var(--accent); background:var(--accent); color:#fff; cursor:pointer; }
  textarea { width:100%; margin-top:10px; min-height:80px; font:13px ui-monospace,Consolas,monospace; background:var(--bg); color:var(--text); border:1px solid var(--border); border-radius:6px; padding:8px; }
</style>
</head>
<body>
<main>
<h1>标注抽查（{{N}} 题）</h1>
<div class="intro">
  <b>不需要医学知识，只需要读懂英文原文。</b>每题都是我改过的一处标注：
  <ol>
    <li>读灰色框里的报告原文，<mark>黄色高亮</mark>是这处修改依据的词句。</li>
    <li>看下面加粗的"改了什么"。</li>
    <li>按"✔ 检查"那一行的问题，在原文里找答案，选同意 / 不同意 / 看不懂。</li>
  </ol>
  全部选完后点最下面的按钮，把生成的结果复制发给我。只要有一题"不同意"，我就会重新检查同一类的所有修改。
</div>
{{CARDS}}
<button id="done">生成结果</button>
<textarea id="result" readonly placeholder="结果会显示在这里"></textarea>
</main>
<script>
document.getElementById("done").onclick = () => {
  const lines = [...document.querySelectorAll(".card")].map((card, i) => {
    const v = card.querySelector("input:checked");
    return `${i + 1}. #${card.dataset.id}: ${v ? v.value : "未选"}`;
  });
  const box = document.getElementById("result");
  box.value = lines.join("\\n");
  box.select();
};
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
