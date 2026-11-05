"""QLoRA supervised fine-tuning for report extraction.

Training data is conversational prompt/completion JSONL where the completion is the label
JSON, so loss is computed on the answer only. Build it with `python -m cre.data.build_sft`.

Usage:
    python -m cre.train_qlora --config configs/qwen2.5-3b-qlora.yaml
"""

import argparse

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        dtype=torch.bfloat16,
        device_map="auto",
    )

    dataset = load_dataset("json", data_files={"train": cfg["train_file"], "dev": cfg["dev_file"]})

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        peft_config=LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            target_modules=cfg["lora"]["target_modules"],
            task_type="CAUSAL_LM",
        ),
        args=SFTConfig(
            output_dir=cfg["output_dir"],
            run_name=cfg.get("run_name"),
            num_train_epochs=cfg["epochs"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            learning_rate=cfg["learning_rate"],
            lr_scheduler_type="cosine",
            warmup_steps=0.05,  # a float < 1 is a ratio of total steps (warmup_ratio was removed)
            max_length=cfg["max_length"],
            gradient_checkpointing=True,
            bf16=True,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            report_to=cfg.get("report_to", "none"),
            seed=cfg.get("seed", 42),
        ),
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
