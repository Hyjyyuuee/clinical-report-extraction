"""Run extraction over a split with an API model or a local (optionally LoRA-adapted) model.

Usage:
    python -m cre.predict --split data/processed/test.jsonl --backend api --model deepseek-chat --few-shot
    python -m cre.predict --split data/processed/test.jsonl --backend local --model Qwen/Qwen2.5-3B-Instruct
    python -m cre.predict --split data/processed/test.jsonl --backend local --model Qwen/Qwen2.5-3B-Instruct --adapter outputs/qwen3b-qlora
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from .prompts import build_messages


def api_generator(model: str):
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ.get("LLM_BASE_URL"))

    def generate(messages: list[dict]) -> str:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    return generate


def local_generator(model_name: str, adapter: str | None, load_in_4bit: bool,
                    constrained: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(adapter or model_name)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    ) if load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quantization, dtype=torch.bfloat16, device_map="auto",
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    new_constraint = None
    if constrained:
        import sys

        import transformers.tokenization_utils_base as tokenization_base

        # lm-format-enforcer 0.11 imports PreTrainedTokenizerBase from a module path that
        # transformers 5 removed; alias it before importing the integration.
        sys.modules.setdefault("transformers.tokenization_utils", tokenization_base)
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_token_enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn,
        )

        from .schema import json_schema

        tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
        schema = json_schema()

        def new_constraint():  # fresh parser state per report
            return build_transformers_prefix_allowed_tokens_fn(tokenizer_data, JsonSchemaParser(schema))

    def generate(messages: list[dict]) -> str:
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(model.device)
        extra = {"prefix_allowed_tokens_fn": new_constraint()} if new_constraint else {}
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=768, do_sample=False, **extra)
        return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--backend", choices=["api", "local"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--few-shot", action="store_true")
    parser.add_argument("--schema-prompt", action="store_true",
                        help="append the explicit JSON format to the system prompt (opt-in)")
    parser.add_argument("--constrained", action="store_true",
                        help="local only: JSON-schema constrained decoding (lm-format-enforcer)")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true",
                        help="keep non-empty outputs already in --out and only run the rest")
    args = parser.parse_args()

    if args.backend == "api":
        generate = api_generator(args.model)
    else:
        generate = local_generator(args.model, args.adapter, not args.no_4bit, args.constrained)
    if args.constrained and args.backend != "local":
        parser.error("--constrained needs --backend local")

    rows = [json.loads(line) for line in args.split.open(encoding="utf-8")][: args.limit]
    tag = Path(args.adapter).name if args.adapter else args.model.split("/")[-1]
    suffix = ("-fewshot" if args.few_shot else "") + ("-schema" if args.schema_prompt else "") \
        + ("-constrained" if args.constrained else "")
    out = args.out or Path("outputs/predictions") / f"{tag}{suffix}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    done = {}
    if args.resume and out.exists():
        for line in out.open(encoding="utf-8"):
            row = json.loads(line)
            if row["raw"]:
                done[row["id"]] = row
        print(f"resume: {len(done)} rows already done")
    todo = [row for row in rows if row["id"] not in done]

    def run(row: dict) -> dict:
        raw = ""
        for attempt in range(args.retries + 1):
            try:
                raw = generate(build_messages(
                    row["report"], few_shot=args.few_shot, schema_prompt=args.schema_prompt,
                ))
                break
            except Exception as error:  # keep going; evaluate counts empty output as invalid
                print(f"\n{row['id']} (attempt {attempt + 1}): {type(error).__name__}: {error}")
                if attempt < args.retries:
                    time.sleep(5 * 2 ** attempt)
        return {"id": row["id"], "raw": raw}

    # API calls are I/O bound, so run them concurrently; a local GPU model runs one at a time.
    # Kept rows are rewritten first and new results are flushed as they finish, so an
    # interrupted run can be continued with --resume.
    workers = args.workers if args.backend == "api" else 1
    with ThreadPoolExecutor(workers) as pool, out.open("w", encoding="utf-8") as f:
        for result in done.values():
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        for result in tqdm(pool.map(run, todo), total=len(todo), desc=tag):
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
