"""Run extraction over a split with an API model or a local (optionally LoRA-adapted) model.

Usage:
    python -m cre.predict --split data/processed/test.jsonl --backend api --model deepseek-chat --few-shot
    python -m cre.predict --split data/processed/test.jsonl --backend local --model Qwen/Qwen2.5-3B-Instruct
    python -m cre.predict --split data/processed/test.jsonl --backend local --model Qwen/Qwen2.5-3B-Instruct --adapter outputs/qwen3b-qlora
"""

import argparse
import json
import os
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


def local_generator(model_name: str, adapter: str | None, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(adapter or model_name)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    ) if load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quantization, torch_dtype=torch.bfloat16, device_map="auto",
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    def generate(messages: list[dict]) -> str:
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(inputs, max_new_tokens=768, do_sample=False)
        return tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True)

    return generate


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--backend", choices=["api", "local"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--few-shot", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.backend == "api":
        generate = api_generator(args.model)
    else:
        generate = local_generator(args.model, args.adapter, not args.no_4bit)

    rows = [json.loads(line) for line in args.split.open(encoding="utf-8")][: args.limit]
    tag = Path(args.adapter).name if args.adapter else args.model.split("/")[-1]
    out = args.out or Path("outputs/predictions") / f"{tag}{'-fewshot' if args.few_shot else ''}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    def run(row: dict) -> dict:
        try:
            raw = generate(build_messages(row["report"], few_shot=args.few_shot))
        except Exception as error:  # keep going; evaluate counts empty output as invalid
            raw = ""
            print(f"\n{row['id']}: {type(error).__name__}: {error}")
        return {"id": row["id"], "raw": raw}

    # API calls are I/O bound, so run them concurrently; a local GPU model runs one at a time.
    workers = args.workers if args.backend == "api" else 1
    with ThreadPoolExecutor(workers) as pool, out.open("w", encoding="utf-8") as f:
        for result in tqdm(pool.map(run, rows), total=len(rows), desc=tag):
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
