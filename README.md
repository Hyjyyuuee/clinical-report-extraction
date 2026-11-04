# Clinical Report Extraction

Extract structured findings — finding type, present / absent / uncertain, laterality, location, and the supporting text span — from free-text chest X-ray reports, and measure how far QLoRA fine-tuning closes the gap between small open-weight models and API models.

## Research questions

1. How do zero-shot, few-shot and QLoRA-fine-tuned open models (Qwen2.5 1.5B / 3B / 7B) compare with an API model on this task?
2. Where do models fail? In particular, negation ("no pleural effusion") and hedging ("pneumonia cannot be excluded").
3. What is the quality / cost / memory trade-off for a model that can run on-premise?

## Data

[IU X-Ray (Open-i)](https://openi.nlm.nih.gov/) — about 3,955 de-identified chest X-ray reports with FINDINGS and IMPRESSION sections, publicly available. Duplicate report texts are removed before splitting so no test report appears in training.

| Split | Size | Labels |
|---|---|---|
| train | rest | Silver: API model few-shot pre-annotation |
| dev | 200 | Gold: silver labels corrected by hand |
| test | 300 | Gold: silver labels corrected by hand |

## Pipeline

```
IU XML -> prepare_iu -> {train,dev,test}.jsonl
train  -> predict (API, few-shot) -> silver -> build_sft -> train_qlora -> LoRA adapter
test   -> predict (API / base / adapter) -> evaluate
```

## Setup

Python 3.11+. PyTorch must be installed first and matched to your GPU; RTX 50-series (Blackwell) cards need a CUDA 12.8+ build:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[train]"
copy .env.example .env
```

## Running

```bash
# 1. Prepare data (after downloading and extracting NLMCXR_reports.tgz into data/raw)
python -m cre.data.prepare_iu --raw data/raw/ecgen-radiology --out data/processed

# 2. Silver labels for training data
python -m cre.predict --split data/processed/train.jsonl --backend api --model deepseek-chat --few-shot --out outputs/predictions/train-silver.jsonl
python -m cre.data.silver --split data/processed/train.jsonl --pred outputs/predictions/train-silver.jsonl --out data/silver/train.jsonl
python -m cre.data.build_sft --labels data/silver/train.jsonl --out data/sft/train.jsonl
python -m cre.data.build_sft --labels data/gold/dev.jsonl --out data/sft/dev.jsonl

# 2b. Correct the API pre-annotations of dev/test by hand (see "Gold labels" below)
python -m cre.predict --split data/processed/test.jsonl --backend api --model deepseek-chat --few-shot --out outputs/predictions/test-preannotation.jsonl
python -m cre.data.silver --split data/processed/test.jsonl --pred outputs/predictions/test-preannotation.jsonl --out data/gold/test.draft.jsonl
python -m cre.review.server

# 3. Fine-tune (3B fits an 8 GB GPU; 7B needs 16 GB+)
python -m cre.train_qlora --config configs/qwen2.5-3b-qlora.yaml

# 4. Predict and evaluate on the gold test set
python -m cre.predict --split data/processed/test.jsonl --backend local --model Qwen/Qwen2.5-3B-Instruct --adapter outputs/qwen2.5-3b-qlora
python -m cre.evaluate --gold data/gold/test.jsonl --pred outputs/predictions/qwen2.5-3b-qlora.jsonl
```

## Gold labels

`python -m cre.review.server` opens a local review page at http://127.0.0.1:8765. It shows each report with every evidence span highlighted by status, and an editable table of findings. Select text in the report and click **Add finding from selected text** to add a missed finding. **Ctrl+Enter** saves and jumps to the next unreviewed report.

Drafts are read from `data/gold/{split}.draft.jsonl`. Only reviewed reports are written to `data/gold/{split}.jsonl`, so evaluation can run on a partially reviewed set.

Annotation rules (also in the system prompt, `src/cre/prompts.py`):

- Label a finding only when the report names it or an unambiguous synonym.
- Do not expand "lungs are clear" or "no acute cardiopulmonary abnormality" into separate absent findings.
- A normal heart or cardiomediastinal silhouette, including "upper limits of normal", is cardiomegaly absent; "stable" alone is not labelled; "borderline" is uncertain.
- "Airspace disease" / "airspace opacity" map to opacity; consolidation only when that word is used.
- Calcified granulomas and prior granulomatous disease are `granuloma`, kept apart from suspicious nodules.
- Hyperinflation is emphysema only when emphysema or COPD is named. Low volumes or crowding are not atelectasis.
- Differentials ("A versus B", "A or B", "may represent", "cannot be excluded") are uncertain. A stated finding with a hedged cause is present.
- Set laterality whenever the side is stated for that finding; "bibasilar" is bilateral, the lingula is left.
- Ignore bones, devices and anything outside the chest. `other` is not scored.

### How the dev and test labels were made

1. DeepSeek (`deepseek-chat`, few-shot) pre-annotated all 300 test and 200 dev reports.
2. Every report was then reviewed against the rules above. The edits are recorded line by line, with a reason for each, in `data/gold/corrections/{test,dev}.txt`, and `python -m cre.data.apply_corrections` rebuilds `data/gold/{split}.jsonl` from the draft and that file. Test: 167 edits in 136 of 300 reports. Dev: 124 edits in 98 of 200 reports. A final pattern-based pass checked that recurring wordings (normal heart size, negated airspace disease) are labelled the same way everywhere.
3. The review was carried out with an LLM assistant (Claude) and spot-checked by hand: 16 test-set edits sampled across all 8 kinds of correction (`python -m cre.review.spot_check`). One was disputed: deleting edema where a report says "vascular congestion" / "heart failure" but never "edema". All 24 atelectasis edits and all 7 congestion-without-edema reports in test and dev were then re-checked and found consistent, and the rule was kept: edema is labelled only when the report names it. Because the drafts came from DeepSeek, DeepSeek's score on this set may be biased upward; report it with that caveat.

## Results

To be filled in after the experiments.

| Model | Setting | Valid JSON | F1 overall | F1 present | F1 absent | F1 uncertain |
|---|---|---|---|---|---|---|
| deepseek-chat | few-shot | | | | | |
| Qwen2.5-3B | zero-shot | | | | | |
| Qwen2.5-3B | QLoRA | | | | | |
| Qwen2.5-7B | QLoRA | | | | | |
