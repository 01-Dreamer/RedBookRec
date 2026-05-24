## Project: RedBookRec

This repository is a recommendation system project based on the local Qilin dataset.

The goal is NOT to build a small demo. The goal is to build a complete, runnable, readable recommendation system project that simulates a Xiaohongshu / REDNote-style content feed recommender.

The project should be developed as a learning-oriented but resume-quality codebase. The user wants to read the code and learn from a complete implementation.

---

## Environment Requirements

The project must use the existing conda environment:

```bash
conda activate rs
````

The environment path is:

```bash
/home/failedman/miniconda3/envs/rs
```

When running commands from scripts or automation, prefer:

```bash
conda run -n rs python <script>
```

Do NOT use the `base` environment.
Do NOT create a new conda environment unless absolutely necessary.
If new packages are needed, install them into the `rs` environment only.

Example:

```bash
conda run -n rs pip install -r requirements.txt
```

---

## Current Repository Structure

Current project root:

```text
RedBookRec/
├── AGENTS.md
└── dataset
    ├── dataset_dict.json
    ├── dqa
    │   └── train-00000-of-00001.parquet
    ├── gitattributes
    ├── notes
    │   ├── train-00000-of-00005.parquet
    │   ├── train-00001-of-00005.parquet
    │   ├── train-00002-of-00005.parquet
    │   ├── train-00003-of-00005.parquet
    │   └── train-00004-of-00005.parquet
    ├── README.md
    ├── recommendation_test
    │   └── train-00000-of-00001.parquet
    ├── recommendation_train
    │   └── train-00000-of-00001.parquet
    ├── search_test
    │   └── train-00000-of-00001.parquet
    ├── search_train
    │   └── train-00000-of-00001.parquet
    └── user_feat
        └── train-00000-of-00001.parquet
```

The dataset is already downloaded locally. Do NOT attempt to download it again.

All data loading should use local parquet files under:

```text
./dataset/
```

---

## Dataset Understanding

The local Qilin dataset contains these logical subsets:

```text
notes                item/content pool
user_feat            user feature table
recommendation_train recommendation training data
recommendation_test  recommendation test data
search_train         search training data
search_test          search test data
dqa                  direct QA / RAG data
```

For the recommendation system, the main subsets are:

```text
recommendation_train
recommendation_test
notes
user_feat
```

`search_train` and `search_test` can be used later for search-intent-enhanced recommendation.

`dqa` is optional and should not be part of the first recommendation pipeline.

---

## Important Development Rule

Do NOT assume column names blindly.

Before building models, implement a schema inspection script that reads all parquet files and saves:

```text
outputs/schema_summary.json
outputs/sample_rows/
```

The project must adapt to the actual field names in the dataset.

Expected inspection script:

```bash
conda run -n rs python scripts/00_inspect_dataset.py
```

The script should print and save:

* file paths
* row counts
* column names
* dtypes
* first few sample rows
* nested/list column summaries

If field names are different from expected, write parser code that maps actual fields to canonical internal names.

Canonical internal names should be:

```text
user_id
note_id
candidate_note_ids
history_note_ids
click
like
collect
comment
share
page_time
position
query
title
content
tags
category
```

Not all fields are guaranteed to exist. The code must handle missing fields gracefully.

---

## Project Goal

Build a complete offline Xiaohongshu-style feed recommendation system:

```text
Qilin local dataset
    ↓
schema inspection
    ↓
data preprocessing
    ↓
user profile construction
    ↓
note profile construction
    ↓
multi-channel recall
    ↓
candidate merge
    ↓
neural ranking model
    ↓
multi-objective scoring
    ↓
reranking
    ↓
offline evaluation
    ↓
recommendation demo / optional Streamlit app
```

The final project should support:

```bash
conda run -n rs python scripts/00_inspect_dataset.py
conda run -n rs python scripts/01_prepare_data.py
conda run -n rs python scripts/02_build_recall.py
conda run -n rs python scripts/03_train_twotower.py
conda run -n rs python scripts/04_train_ranker.py
conda run -n rs python scripts/05_evaluate.py
conda run -n rs python scripts/06_recommend_user.py --user_id <some_user_id>
```

Optional:

```bash
conda run -n rs streamlit run app.py
```

---

## Expected Final Repository Structure

Create a clean project structure:

```text
RedBookRec/
├── AGENTS.md
├── README.md
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── recall.yaml
│   ├── twotower.yaml
│   └── ranker.yaml
├── dataset/
│   └── ...
├── data_processed/
│   ├── schema/
│   ├── mappings/
│   ├── features/
│   ├── samples/
│   └── recalls/
├── models/
│   ├── twotower/
│   └── ranker/
├── outputs/
│   ├── schema_summary.json
│   ├── metrics/
│   ├── recommendations/
│   └── figures/
├── scripts/
│   ├── 00_inspect_dataset.py
│   ├── 01_prepare_data.py
│   ├── 02_build_recall.py
│   ├── 03_train_twotower.py
│   ├── 04_train_ranker.py
│   ├── 05_evaluate.py
│   └── 06_recommend_user.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── io.py
│   │   ├── schema.py
│   │   ├── parser.py
│   │   ├── preprocess.py
│   │   └── dataset.py
│   ├── features/
│   │   ├── user_features.py
│   │   ├── note_features.py
│   │   └── sequence_features.py
│   ├── recall/
│   │   ├── popular.py
│   │   ├── itemcf.py
│   │   ├── text_recall.py
│   │   ├── twotower_recall.py
│   │   └── merge.py
│   ├── models/
│   │   ├── layers.py
│   │   ├── two_tower.py
│   │   ├── din_ranker.py
│   │   └── losses.py
│   ├── train/
│   │   ├── train_twotower.py
│   │   └── train_ranker.py
│   ├── eval/
│   │   ├── metrics.py
│   │   └── evaluator.py
│   ├── rerank/
│   │   ├── diversity.py
│   │   └── rules.py
│   └── service/
│       └── recommender.py
└── app.py
```

---

## Modeling Requirements

Do not make the project a simple LightGBM-only demo.

The main learning value should come from a neural recommendation pipeline.

The project should include:

### 1. Multi-channel Recall

Implement at least:

```text
PopularRecall
ItemCFRecall
TextRecall if text fields are available
TwoTowerRecall
```

For `TextRecall`, use BM25 or TF-IDF first. If dependencies are available, optionally support sentence embeddings later.

For `TwoTowerRecall`, use PyTorch.

TwoTower design:

```text
User tower:
    recent history note ids -> embedding -> pooling/attention -> user vector

Item tower:
    note id -> embedding
    optional text/category features -> embedding
    final item vector

Training:
    in-batch negatives or sampled negatives
    contrastive / softmax loss
```

The code should be understandable and not overly abstract.

### 2. Neural Ranking Model

Implement a DIN-style ranker or DeepFM/DIN hybrid ranker in PyTorch.

Preferred ranker:

```text
DINRanker
```

Inputs:

```text
user id
target note id
user history note sequence
target note features
position/context features
recall source features
```

DIN-style behavior:

```text
target note embedding attends over user's historical note embeddings
attention output represents user interest for the target note
MLP predicts click probability
```

Support multi-task outputs if labels are available:

```text
click
like
collect
comment
share
page_time
```

If some labels are missing, fall back gracefully to click-only training.

Multi-task loss:

```text
BCE for click/like/collect/comment/share
SmoothL1 or MSE for normalized page_time
```

### 3. Multi-objective Scoring

If labels exist, define final recommendation score as:

```text
score =
    0.45 * P(click)
  + 0.25 * P(collect)
  + 0.15 * P(like)
  + 0.05 * P(comment)
  + 0.05 * P(share)
  + 0.05 * normalized_page_time_score
```

If some labels are unavailable, normalize the weights over available targets.

### 4. Reranking

Implement simple rule-based reranking:

```text
author dedup if author field exists
category diversity if category field exists
tag diversity if tag field exists
history filtering
score-based top-k reranking
```

The reranker must never crash if author/category/tag fields do not exist.

---

## Evaluation Requirements

Implement offline evaluation metrics.

At minimum:

```text
AUC if binary labels exist
LogLoss if click labels exist
Recall@K
NDCG@K
HitRate@K
MRR@K
coverage
average recommendation score
```

Evaluate:

```text
PopularRecall baseline
ItemCFRecall
TwoTowerRecall
DINRanker
DINRanker + Rerank
```

Save metrics to:

```text
outputs/metrics/
```

Recommended output file:

```text
outputs/metrics/eval_summary.json
outputs/metrics/eval_summary.csv
```

---

## Recommendation Demo Requirements

Implement:

```bash
conda run -n rs python scripts/06_recommend_user.py --user_id <id>
```

The output should show:

```text
user_id
user profile summary
recent history notes if available
Top-N recommended notes
note title/content snippet if available
score
recall source
rerank reason
```

Also support:

```bash
conda run -n rs python scripts/06_recommend_user.py --random_user
```

This should pick a valid user from processed data and display recommendations.

---

## Optional Streamlit App

If time allows, create:

```text
app.py
```

It should support:

```bash
conda run -n rs streamlit run app.py
```

The app should show:

```text
user selector
user profile
history notes
recommended notes
score breakdown
recall source
rerank explanation
```

The app is optional. The command-line recommendation demo is required.

---

## Data Safety and Storage Rules

Do NOT modify files inside:

```text
./dataset/
```

All generated files should go into:

```text
./data_processed/
./models/
./outputs/
```

Do not commit huge generated artifacts unless necessary.

Do not duplicate the entire dataset.

Do not convert all large parquet files to CSV unless absolutely necessary.

Use parquet for processed data.

---

## Performance Requirements

The project should be runnable on a normal local machine.

Use config options to limit data size during development:

```yaml
debug:
  enabled: true
  max_users: 5000
  max_notes: 50000
  max_interactions: 200000
```

Also support full mode:

```yaml
debug:
  enabled: false
```

Default mode should be development-friendly and runnable within reasonable time.

Use batching for PyTorch models.

Use CPU by default, but automatically use CUDA if available.

---

## Dependency Requirements

Create `requirements.txt`.

Likely dependencies:

```text
pandas
numpy
pyarrow
scikit-learn
torch
tqdm
pyyaml
rich
rank-bm25
scipy
streamlit
```

Optional dependencies:

```text
faiss-cpu
sentence-transformers
```

If optional dependencies are not installed, code should skip optional modules gracefully.

Install into rs environment:

```bash
conda run -n rs pip install -r requirements.txt
```

---

## Coding Style

Use clear, readable Python.

Prefer explicit code over overly clever abstractions.

Use type hints where useful.

Add comments explaining recommendation logic.

Do not write notebook-only code. The project should be script-based and runnable.

Use deterministic random seeds:

```python
seed = 2025
```

Use logging instead of excessive print statements where possible.

Every script should have a `main()` function.

Every long-running script should show progress with `tqdm`.

---

## Robustness Requirements

The dataset schema may contain nested lists, dict-like fields, or unexpected column names.

Therefore:

1. Implement schema inspection first.
2. Implement field mapping logic.
3. Never hard-code a single possible column name without fallback.
4. If a required field is missing, raise a clear error message explaining which field is missing and how to inspect it.
5. If an optional feature is missing, skip it and continue.

---

## README Requirements

Create a strong `README.md` explaining:

```text
project background
dataset description
system architecture
module design
how to run
model details
evaluation metrics
sample recommendation output
future work
```

The README should emphasize:

```text
Qilin is a Xiaohongshu / REDNote-style public research dataset.
This project is an offline content recommendation system.
It is not a reproduction of the real Xiaohongshu production system.
```

Do not claim to reproduce Xiaohongshu's real online recommender.

Correct wording:

```text
A Xiaohongshu-style content feed recommendation system based on the public Qilin dataset.
```

Avoid incorrect wording:

```text
Reproducing Xiaohongshu's real recommendation algorithm.
```

---

## Minimal Successful Deliverable

The minimal acceptable final deliverable is:

```text
1. Dataset inspection works.
2. Data preprocessing works.
3. At least two recall methods work.
4. A PyTorch TwoTower recall model trains.
5. A PyTorch DIN-style ranker trains.
6. Offline evaluation runs and saves metrics.
7. User recommendation demo works.
8. README explains the full system clearly.
```

The final code should be runnable with:

```bash
conda run -n rs pip install -r requirements.txt

conda run -n rs python scripts/00_inspect_dataset.py
conda run -n rs python scripts/01_prepare_data.py
conda run -n rs python scripts/02_build_recall.py
conda run -n rs python scripts/03_train_twotower.py
conda run -n rs python scripts/04_train_ranker.py
conda run -n rs python scripts/05_evaluate.py
conda run -n rs python scripts/06_recommend_user.py --random_user
```

---

## Development Priority

Build in this order:

1. Schema inspection
2. Canonical parser
3. Data preprocessing
4. Popular recall
5. ItemCF recall
6. TwoTower training
7. Candidate merge
8. DIN ranker
9. Evaluation
10. Recommendation demo
11. Reranking
12. Streamlit app
13. README polish

Do not start with the UI.
Do not start with advanced embeddings.
Do not get stuck on optional image features.

The core value is a complete recommendation system pipeline.

````

我建议你保存后，对 Codex 的第一句话可以这样说：

```text
请严格阅读 AGENTS.md，基于当前本地 dataset 目录开发 RedBookRec。不要做 demo，要实现一个可运行的完整推荐系统项目。所有命令使用 conda run -n rs。
````
