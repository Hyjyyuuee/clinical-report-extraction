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

# 4b. Fairer prompting baselines for the base model (opt-in; the default prompt is unchanged)
#     --schema-prompt adds the explicit JSON format; --constrained (local only) adds
#     JSON-schema constrained decoding via lm-format-enforcer (pip install lm-format-enforcer)
python -m cre.predict --split data/processed/test.jsonl --backend local --model Qwen/Qwen2.5-3B-Instruct --schema-prompt
python -m cre.predict --split data/processed/test.jsonl --backend local --model Qwen/Qwen2.5-3B-Instruct --few-shot --schema-prompt --constrained
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

Gold test set: 300 reports, 1,088 scored findings (174 present / 861 absent / 53 uncertain). A finding matches on (finding, status, laterality). `cre.evaluate` reports two metrics:

- **Strict** (the original metric, unchanged): the whole output must parse and validate against the schema. If it does not, the report contributes no predictions, so all of its gold findings count as misses. "Valid JSON" is the share of reports that pass.
- **Lenient**: the output only needs to be a JSON object with a `findings` list. Each finding that fails validation (an unknown finding name, a missing `status`, …) is dropped on its own, and the remaining findings are scored. "Parseable" is the share of reports with such a list. "Dropped" is the number of findings discarded across all 300 reports.

Prompt settings: *zero-shot* / *few-shot* use the default prompt, which is also what DeepSeek used and what the QLoRA adapter was trained on. *+ schema prompt* (`--schema-prompt`) adds the full JSON field format, a minimal example, and an instruction to choose `finding` only from the list. *+ constrained* (`--constrained`) also restricts decoding to the JSON schema with lm-format-enforcer.

| Model | Setting | Valid JSON | F1 overall | F1 present | F1 absent | F1 uncertain | Parseable | Dropped | Lenient F1 overall | Lenient F1 present | Lenient F1 absent | Lenient F1 uncertain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-chat | few-shot | 100.0% | 0.959* | 0.926* | 0.974* | 0.822* | 100.0% | 0 | 0.959* | 0.926* | 0.974* | 0.822* |
| Qwen2.5-3B | zero-shot | 13.0% | 0.000 | 0.000 | 0.000 | 0.000 | 78.7% | 743 | 0.000 | 0.000 | 0.000 | 0.000 |
| Qwen2.5-3B | few-shot | 20.0% | 0.212 | 0.147 | 0.232 | 0.073 | 92.0% | 512 | 0.599 | 0.538 | 0.628 | 0.242 |
| Qwen2.5-3B | zero-shot + schema prompt | 29.7% | 0.157 | 0.126 | 0.170 | 0.037 | 81.7% | 373 | 0.366 | 0.367 | 0.382 | 0.036 |
| Qwen2.5-3B | few-shot + schema prompt | 27.7% | 0.215 | 0.194 | 0.229 | 0.037 | 94.3% | 430 | 0.515 | 0.565 | 0.521 | 0.188 |
| Qwen2.5-3B | zero-shot + schema prompt + constrained | 99.0% | 0.485 | 0.525 | 0.491 | 0.192 | 99.0% | 0 | 0.485 | 0.525 | 0.491 | 0.192 |
| Qwen2.5-3B | few-shot + schema prompt + constrained | 99.7% | 0.599 | 0.691 | 0.595 | 0.240 | 99.7% | 0 | 0.599 | 0.691 | 0.595 | 0.240 |
| Qwen2.5-3B | QLoRA | 99.7% | 0.947 | 0.884 | 0.972 | 0.743 | 100.0% | 1 | 0.948 | 0.884 | 0.974 | 0.743 |
| Qwen2.5-7B | QLoRA | — | — | — | — | — | — | — | — | — | — | — |

\* The gold labels started as DeepSeek pre-annotations and were then corrected, so DeepSeek's score is likely biased upward: wherever a draft label was left unchanged, DeepSeek agrees with the gold label by construction.

Constrained decoding runs at about 12 s per report on the 8 GB laptop GPU, against about 3 s unconstrained. lm-format-enforcer 0.11 imports a module path that transformers 5 removed; `predict.py` adds an alias for it. The few constrained outputs that are still invalid ran into the 768-token generation limit while repeating findings.

Setup: QLoRA r=16 on all attention and MLP projections, 3 epochs over 2,658 silver training reports (one of the 2,659 DeepSeek outputs could not be parsed), 4-bit NF4. It took 2 h 23 min on an RTX 5060 Laptop GPU (8 GB). Dev loss was 0.0525 / 0.0475 / 0.0529 after epochs 1 / 2 / 3; the adapter in `outputs/qwen2.5-3b-qlora` is from epoch 3. Producing the silver labels took 2,659 `deepseek-chat` calls, under ¥10. **The 7B run was not done**: the README estimate for 7B is 16 GB+, and this machine has 8 GB (`configs/qwen2.5-7b-qlora.yaml` is ready for a larger GPU). Metrics are in `outputs/eval-*.json` and predictions in `outputs/predictions/`.

### Error analysis

- **Without fine-tuning, the 3B model mostly fails on the format, not on negation.** In zero-shot, 134/300 outputs use finding names outside the allowed list, 63 are malformed JSON, and 60 omit `status`. The zero-shot prompt names the allowed values but never shows the field schema, and one invalid finding invalidates the whole report. Few-shot fixes the missing fields, but 212/300 outputs still invent finding types (e.g. `indistinct_vascular_margination`). QLoRA brings valid output to 299/300; the remaining failure is an invented `pleural_thickening`.
- **Telling the model the schema helps little; constraining decoding helps a lot, but neither comes close to fine-tuning.** With the schema prompt, strict validity rises only from 13% to 30% (zero-shot), and the 3B model still invents finding names (373 dropped in zero-shot, e.g. `calcified_granuloma`, `degenerative_changes`, `low_lung_volumes`, or `null`). In few-shot, adding the schema prompt even lowers lenient F1 (0.599 → 0.515). Constrained decoding guarantees valid output and gives the best base-model scores (0.485 zero-shot, 0.599 few-shot). These are still far below QLoRA (0.947). The gap is now mostly recall (few-shot constrained: precision 0.730, recall 0.508). The base model leaves out most routine negations: of 861 gold absent findings, 134 cardiomegaly-absent, 96 pleural-effusion-absent and 84 pneumothorax-absent are missed. It also labels hedges as present (23 uncertain → present). So the lenient metric shows that strict scoring exaggerates the base model's failure, at 0.21 vs 0.60 for few-shot. But once the format is fixed, what remains is a gap in following the annotation guidelines, not in formatting.
- **Negation is close to solved.** After QLoRA, absent F1 is 0.972, against 0.974 for DeepSeek. Only one gold-absent finding was predicted present. Most remaining absent errors are boundary cases rather than flipped polarity: a negated "airspace disease" (opacity absent) is missed (10–11 misses for each model), and extra or missed cardiomegaly-absent labels on borderline heart-size wording.
- **Hedging is the main remaining gap.** Uncertain is the weakest class for every model (QLoRA 0.743 vs DeepSeek 0.822). The dominant error for both is collapsing a hedge into present (QLoRA 12, DeepSeek 9), e.g. "scarring or atelectasis", "findings suggestive of emphysema". The reverse also occurs, where a stated finding with a hedged cause is marked uncertain (QLoRA 5, DeepSeek 4), e.g. "opacities, atelectasis versus airspace disease" should be opacity present. The training labels are DeepSeek outputs, so the student inherits the teacher's hedging errors. With only 53 uncertain gold findings, these F1 differences are noisy.
