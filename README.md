# tt501 at ArchEHR-QA 2026

Code and prompt templates for the system submitted by team **tt501** to the
[ArchEHR-QA 2026 shared task](https://archehr-qa.github.io/) (CL4Health @ LREC 2026).

**Paper:** *tt501 at ArchEHR-QA 2026: Few-Shot Prompting with Retrieval-Augmented
Generation for Grounded Clinical EHR Question Answering*

---

## Repository Structure

```
tt501-archehr-qa-2026/
├── subtask2/
│   ├── hybrid_subtask2.py          # BM25 retrieval utilities (shared)
│   ├── subtask2_ensemble_refine.py # Run 2: full-context CoT + refinement (submitted)
│   └── eval_subtask2.py            # Evaluation script (strict/lenient F1)
├── subtask3/
│   ├── subtask3_zeroshot_v1.py     # Run 1: zero-shot, full note, no examples (baseline)
│   ├── subtask3_rag_fewshot.py     # Run 2: RAG + few-shot generation (submitted)
│   └── eval_subtask3.py            # Evaluation script
├── subtask4/
│   ├── subtask4_fewshot_v1.py      # Run 1: basic few-shot alignment
│   ├── subtask4_fewshot_v2.py      # Run 2: recall-optimised + reasoning annotations (submitted)
│   └── eval_subtask4.py            # Evaluation script (micro/macro F1)
└── fewshot_examples.txt            # Few-shot demonstrations used in Subtask 3
```

---

## Results

| Subtask | Run | Cases | API Mode | Approx. Time | Best Metric | Score |
|---|---|---|---|---|---|---|
| 2 | hybrid-bm25-llm | 47 | Synchronous | ~3 min | Strict Micro-F1 | |
| 2 | ensemble-refine-v1 | 47 | Async Batch | ~10 min | Strict Micro-F1 | **58.8** |
| 3 | zero-shot-v1 | 47 | Async Batch | ~10 min | Overall | |
| 3 | rag-fewshot-v1 | 47 | Async Batch | ~10 min | Overall | **31.4** |
| 4 | fewshot-v1 | 147 | Async Batch | ~15 min | Micro-F1 | |
| 4 | fewshot-v2 | 147 | Async Batch | ~15 min | Micro-F1 | **79.1** |

Bold scores are from the officially submitted runs.

---

## Setup

```bash
pip install openai requests rank_bm25 numpy
```

Set your xAI API key:

```bash
cp .env.example .env
# then edit .env and fill in your key
```

Or export directly:

```bash
export XAI_API_KEY=your_key_here
```

---

## Usage

All scripts use the [xAI Batch API](https://docs.x.ai/developers/advanced-api-usage/batch-api).
The pipeline for each subtask is: **upload** → monitor → **status** → **download**.

After uploading, you can track progress in real time from the
[xAI Console → Batches](https://console.x.ai/team/default/batches) page — it shows
estimated completion time and per-request status without polling the CLI.
Run the `status` command (or refresh the console) until the batch shows `completed`,
then run `download`.

**Subtask 2 — Evidence Identification:**
```bash
# Step 1: create batch and upload requests
python subtask2/subtask2_ensemble_refine.py upload --mode test

# Step 2: check status (repeat until complete)
python subtask2/subtask2_ensemble_refine.py status --batch-id <BATCH_ID>

# Step 3: download and save predictions
python subtask2/subtask2_ensemble_refine.py download --batch-id <BATCH_ID> --mode test

# Optional: self-consistency run (5 samples + majority voting)
python subtask2/subtask2_ensemble_refine.py upload --mode test --samples 5
python subtask2/subtask2_ensemble_refine.py download --batch-id <BATCH_ID> --mode test --voting
```

**Subtask 3 — Answer Generation:**

Run 1 (zero-shot baseline, uses full note directly):
```bash
python subtask3/subtask3_zeroshot_v1.py upload
python subtask3/subtask3_zeroshot_v1.py status --batch-id <BATCH_ID>
python subtask3/subtask3_zeroshot_v1.py download --batch-id <BATCH_ID>
```

Run 2 (RAG + few-shot, requires Subtask 2 output first):
```bash
python subtask3/subtask3_rag_fewshot.py upload
python subtask3/subtask3_rag_fewshot.py status --batch-id <BATCH_ID>
python subtask3/subtask3_rag_fewshot.py download --batch-id <BATCH_ID>
```

**Subtask 4 — Evidence Alignment (Run 2, submitted):**
```bash
python subtask4/subtask4_fewshot_v2.py upload --split test
python subtask4/subtask4_fewshot_v2.py status --batch-id <BATCH_ID>
python subtask4/subtask4_fewshot_v2.py download --batch-id <BATCH_ID> --split test
```

**Dev evaluation:**
```bash
python subtask2/eval_subtask2.py
python subtask3/eval_subtask3.py
python subtask4/eval_subtask4.py
```

---

## Data

The ArchEHR-QA 2026 dataset is derived from the
[MIMIC database](https://physionet.org/content/mimiciii/) and is **not publicly
redistributable**. Data files are therefore **not** included in this repository.

To obtain the data:

1. Complete the [PhysioNet credentialing process](https://physionet.org/register/) and sign the MIMIC Data Use Agreement.
2. Register for the [ArchEHR-QA 2026 shared task](https://archehr-qa.github.io/) — participants receive the dataset directly from the task organisers after registration.
3. Place the downloaded files under `data/` following this structure:

```
data/
├── v1.4/
│   ├── dev/
│   │   ├── archehr-qa.xml
│   │   ├── archehr-qa_key.json
│   │   └── archehr-qa_mapping.json
│   └── test-2026/
│       ├── archehr-qa.xml
│       ├── archehr-qa_key.json
│       └── archehr-qa_mapping.json
└── subtask4/
    └── v1.5/
        ├── dev/
        └── test-2026/
```


