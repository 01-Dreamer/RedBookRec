# AGENTS.md

## Project Goal

RedBookRec is a Xiaohongshu-style recommendation system built on the Qilin dataset. The target architecture is a four-stage industrial recommendation pipeline: Two-Tower recall, DCN-lite pre-ranking, SIM fine-ranking, and DPP re-ranking. The first milestone is a runnable version of this pipeline with data preparation, offline evaluation, and a simple demo app.

## Runtime Environment

Use the existing `rs` conda environment for this project.

```bash
conda activate rs
```

The expected prompt is:

```text
(rs) failedman@Ubuntu:~/RedBookRec$
```

When running commands from automation or scripts where shell activation is inconvenient, prefer:

```bash
conda run -n rs python ...
conda run -n rs streamlit run app.py
```

Avoid installing dependencies into `base`. If new packages are needed, install them into `rs` and record them in `requirements.txt`.

## Local Hardware and Run Modes

The current development machine has about 16 CPU cores and 24 GB memory. Use it for smoke tests, schema checks, small-sample preprocessing, and short training runs only. Do not require full-dataset training to pass during local development.

All data and training scripts must provide CPU/GPU-friendly runtime controls:

- `--debug`: run a small, fast sample for local validation.
- `--device cpu|cuda|auto`: choose CPU, GPU, or automatic device selection.
- `--max-users`: limit the number of users during local tests.
- `--max-notes`: limit the number of notes during local tests.
- `--max-interactions`: limit the number of interaction rows during local tests.
- `--batch-size`: allow smaller CPU batches and larger GPU batches.
- `--num-workers`: allow safe dataloader parallelism; default should not exceed local CPU capacity.
- `--epochs`: allow smoke-test runs with 1 epoch.
- `--mixed-precision`: optional GPU-only acceleration; must be disabled automatically on CPU.

Default behavior should be development-friendly and runnable on CPU. Full training should be explicitly requested through config or command-line flags.

Example local smoke-test commands:

```bash
conda run -n rs python scripts/prepare_data.py --debug --max-users 2000 --max-notes 50000 --max-interactions 200000
conda run -n rs python scripts/build_search_index.py --debug --max-notes 50000
conda run -n rs python scripts/train_twotower.py --debug --device cpu --epochs 1 --batch-size 512
conda run -n rs python scripts/train_dcn_lite.py --debug --device cpu --epochs 1 --batch-size 512
conda run -n rs python scripts/train_sim.py --debug --device cpu --epochs 1 --batch-size 256
conda run -n rs python scripts/evaluate.py --debug --top-k 20
```

Example GPU-server commands:

```bash
conda run -n rs python scripts/prepare_data.py --full
conda run -n rs python scripts/build_search_index.py --full
conda run -n rs python scripts/train_twotower.py --full --device cuda --epochs 10 --batch-size 4096 --mixed-precision
conda run -n rs python scripts/train_dcn_lite.py --full --device cuda --epochs 10 --batch-size 4096 --mixed-precision
conda run -n rs python scripts/train_sim.py --full --device cuda --epochs 5 --batch-size 1024 --mixed-precision
conda run -n rs python scripts/evaluate.py --full --top-k 100
```

## Global Configuration

Use a unified config system for the whole project. Do not scatter hard-coded paths, batch sizes, device choices, sample limits, model dimensions, or SIM parameters across scripts.

Recommended config files:

```text
configs/
  base.yaml          shared defaults for all stages
  debug.yaml         local smoke-test overrides
  full.yaml          full-data training overrides
  recall.yaml        search recall and Two-Tower settings
  rank.yaml          DCN-lite and SIM settings
  rerank.yaml        DPP settings
```

Every script should accept:

- `--config`: one or more YAML config files.
- `--debug`: shortcut for loading debug-friendly limits.
- `--full`: shortcut for full-data mode.
- command-line overrides for important runtime fields, such as `--device`, `--batch-size`, `--epochs`, `--max-users`, `--max-notes`, `--max-interactions`, `--sim-last-n`, and `--sim-top-k`.

Precedence order:

```text
base.yaml
  -> stage config, such as recall.yaml or rank.yaml
  -> debug.yaml or full.yaml
  -> explicit command-line overrides
```

The merged config should be saved with each run under `artifacts/runs/<run_id>/config.yaml` so results can be reproduced.

The config must cover at least:

- paths: raw dataset, processed data, artifacts, indexes, checkpoints, metrics, logs,
- runtime: seed, mode, device, num workers, batch size, epochs, mixed precision,
- debug/full limits: max users, max notes, max interactions, max history length,
- ID mapping: reserved default IDs and unknown handling,
- search recall: text fields, index type, top K, tokenizer options,
- Two-Tower: embedding dimension, hidden sizes, negative sampling, ANN settings,
- DCN-lite: feature groups, cross layers, MLP layers, dropout,
- SIM: `lastN`, `topK`, max history, history behavior types, attention dimensions,
- DPP: final top K, diversity weight, similarity feature type,
- evaluation: metrics, K values, split names, output paths.

Example commands:

```bash
conda run -n rs python scripts/prepare_data.py --config configs/base.yaml configs/debug.yaml
conda run -n rs python scripts/train_twotower.py --config configs/base.yaml configs/recall.yaml configs/debug.yaml --device cpu
conda run -n rs python scripts/train_sim.py --config configs/base.yaml configs/rank.yaml configs/debug.yaml --sim-last-n 20 --sim-top-k 20
conda run -n rs python scripts/evaluate.py --config configs/base.yaml configs/debug.yaml
```

## Dataset Layout

The local Qilin dataset is under `dataset/`.

Important splits:

- `dataset/recommendation_train/`: training logs for recommendation ranking.
- `dataset/recommendation_test/`: test logs for offline evaluation.
- `dataset/notes/`: note corpus with title, content, category, engagement, and media metadata.
- `dataset/user_feat/`: user profile and dense user features.
- `dataset/search_train/` and `dataset/search_test/`: search behavior, useful for search-aware recommendation later.
- `dataset/dqa/`: question-answering/RAG data, useful in later explanation or assistant features.

## Target Recommendation Pipeline

Use the following four-stage design for Qilin-based RedBookRec:

| Stage | Algorithm | Role |
| --- | --- | --- |
| Recall | Search Recall + Two-Tower | Use keyword search when the user provides a query, otherwise retrieve personalized feed candidates from the large note corpus. |
| Pre-rank | DCN-lite | Use lightweight feature crossing to filter weak candidates efficiently. |
| Fine-rank | SIM | Use long user behavior history to estimate detailed interest in each candidate note. |
| Re-rank | DPP | Select a high-quality and diverse final recommendation list from high-score candidates. |

This design is suitable for Qilin because the dataset contains note metadata, user features, recent clicked notes, recommendation exposure logs, click and engagement labels, query context, search logs, and timestamps. For SIM, build longer user behavior sequences from `recommendation_train` by grouping clicked or engaged notes by `user_idx` and sorting by request or interaction timestamp.

## SIM Fine-Ranking Design

Use SIM as the fine-ranking model after DCN-lite. SIM must explicitly combine recent behavior and long-history target-aware retrieval:

1. `lastN` recent behavior sequence
   - Keep the most recent `N` clicked or engaged notes for each user.
   - This branch captures short-term interest and session-level preference.
   - Suggested local default: `lastN = 20`.
   - Suggested full-training range: `lastN = 50` to `100`.

2. `topK` target-aware long-history sequence
   - Build a longer user behavior sequence from `recommendation_train`.
   - For each candidate note, retrieve the top `K` historical notes most similar to the candidate.
   - Similarity can start with note embedding dot product, taxonomy overlap, or text embedding similarity.
   - This branch captures long-term but candidate-relevant interest.
   - Suggested local default: `topK = 20`.
   - Suggested full-training range: `topK = 50` to `100`.

3. Candidate-aware interest modeling
   - Use the candidate note embedding as the target query.
   - Apply attention over both `lastN` and `topK` behavior embeddings.
   - Concatenate target note features, user features, context features, `lastN` interest vector, and `topK` interest vector.
   - Feed the merged vector into an MLP to predict click or multi-task engagement scores.

4. Sequence defaults and robustness
   - If the user history is shorter than `lastN`, pad with `note_id = 0`.
   - If long-history retrieval returns fewer than `topK`, pad with `note_id = 0`.
   - If the user is new or unknown, both sequences should be empty/padded and the model should rely on default user features, context, and candidate features.

SIM-related scripts must expose these parameters:

- `--sim-last-n`: number of most recent behavior notes.
- `--sim-top-k`: number of target-aware long-history notes.
- `--sim-max-history`: maximum stored behavior length per user before topK retrieval.
- `--sim-history-types`: behavior types used to build history, such as click, like, collect, comment, and share.

Example local SIM command:

```bash
conda run -n rs python scripts/train_sim.py --debug --device cpu --epochs 1 --batch-size 256 --sim-last-n 20 --sim-top-k 20 --sim-max-history 200
```

Example GPU SIM command:

```bash
conda run -n rs python scripts/train_sim.py --full --device cuda --epochs 5 --batch-size 1024 --mixed-precision --sim-last-n 50 --sim-top-k 100 --sim-max-history 1000
```

The recall stage must support two request modes:

1. Search recommendation mode
   - Triggered when the user provides a non-empty keyword or query.
   - Use keyword-driven recall as the primary channel.
   - Return top candidate notes matching the query from title, content, taxonomy, or precomputed text index.
   - Start with BM25 or TF-IDF, then optionally add dense query-note retrieval using DPR-style embeddings.
   - Qilin `search_train`, `search_test`, `bm25_results`, and `dpr_results` can be used to validate and improve this mode.

2. Non-search feed recommendation mode
   - Triggered when the user does not provide a query.
   - Use Two-Tower personalized recall as the primary channel.
   - Use user profile, dense user features, recent clicks, reconstructed long behavior sequence, and note metadata.
   - Optional fallback channels can include Popular, ItemCF, and category-based recall.

## Recommended MVP

Build the first version in four layers:

1. Data preparation
   - Read Qilin parquet files.
   - Expand `rec_result_details_with_idx` into one row per exposed note.
   - Join note metadata from `notes` and user metadata from `user_feat`.
   - Create labels such as `click_label` and `engage_label`.
   - Reconstruct user behavior sequences for Two-Tower and SIM.
   - Build stable ID mappings for users and notes with reserved default IDs for missing, unknown, or new entities.

2. Recall
   - Implement search recall for keyword-driven recommendation.
   - Implement Two-Tower retrieval with a user tower and a note tower for non-search feed recommendation.
   - User tower inputs: user profile, dense user features, query context, and recent or reconstructed clicked-note sequence.
   - Note tower inputs: note ID, taxonomy, note type, text features, and engagement statistics.
   - Train with clicked or engaged notes as positives and exposed-but-not-clicked or sampled notes as negatives.
   - At request time, route non-empty queries to search recall and empty-query requests to Two-Tower recall.

3. Pre-rank and fine-rank
   - Use DCN-lite for fast feature crossing over user, item, context, and recall-source features.
   - Use SIM for fine-ranking with long user behavior history and candidate-aware interest matching.
   - SIM must use both `lastN` recent behaviors and `topK` target-aware long-history behaviors.
   - Keep model outputs calibrated enough for downstream re-ranking.

4. Re-rank and demo app
   - Apply DPP to balance score quality and diversity.
   - Use note taxonomy or text embeddings as item similarity features for DPP.
   - Implement `app.py` as the first user-facing entrypoint.
   - Input: `user_idx`, optional query/context, and `top_k`.
   - Output: recommended notes with title, content snippet, category, score, and a short reason.

## Offline Evaluation

Evaluate each stage separately and the full pipeline end-to-end.

1. Recall evaluation
   - Evaluate whether Two-Tower candidates cover clicked or engaged notes.
   - Evaluate whether search recall covers clicked or engaged notes for query requests.
   - Track `Recall@50`, `Recall@100`, `Recall@200`, and `Recall@500`.

2. Ranking evaluation
   - Evaluate on `recommendation_test`.
   - Track `Recall@K`, `HitRate@K`, `MRR@K`, and `NDCG@K`.
   - For CTR-style labels, also track `AUC` and `LogLoss`.

3. Re-ranking evaluation
   - Track final list quality and diversity together.
   - Use `NDCG@10`, `HitRate@10`, category coverage, intra-list diversity, and duplicate/category concentration checks.

## ID Mapping and Cold-Start Defaults

The project must handle missing IDs, unseen users, unseen notes, and newly created entities without crashing.

Reserve default IDs in every categorical mapping:

| Entity | Reserved ID | Meaning |
| --- | --- | --- |
| User | `0` | Unknown, missing, or new user. |
| Note | `0` | Unknown, missing, or new note. |
| Category/taxonomy | `0` | Unknown or missing category. |
| Query/token bucket | `0` | Unknown or empty query token. |

Real dataset IDs should start from `1` after mapping. Never use raw `user_idx` or `note_idx` directly as embedding indices unless the mapping explicitly reserves `0`.

Required behavior:

- If `user_idx` is missing or not found in the user mapping, map it to `user_id = 0`.
- If `note_idx` is missing or not found in the note mapping, map it to `note_id = 0`.
- If a user has no history, use an empty sequence padded with `note_id = 0`.
- If a note has missing text, category, or statistics, fill with safe defaults instead of dropping the request.
- If a new user arrives in non-search feed mode, Two-Tower should use `user_id = 0` plus available context and fall back to popular/category/search-derived candidates when needed.
- If a new note arrives, use `note_id = 0` plus available content/category/statistical features until the ID mapping and embeddings are refreshed.

Cold-start fallback policy:

1. New user with query
   - Use search recall from the keyword.
   - Apply DCN-lite/SIM with default user ID and available context.

2. New user without query
   - Use popular, category, or location/context fallback recall.
   - Use default user ID and empty history.

3. New note
   - Use default note ID.
   - Score with available note text, taxonomy, and statistics.
   - Allow it into recall only through search, category, popular/new-item exploration, or refreshed ANN indexes.

4. Missing IDs in training or evaluation
   - Map to reserved defaults.
   - Log counts of unknown users, unknown notes, and missing categories.
   - Do not silently discard rows unless the label itself is unusable.

## Suggested Project Structure

```text
RedBookRec/
  app.py
  requirements.txt
  AGENTS.md
  configs/
    base.yaml
    debug.yaml
    full.yaml
    recall.yaml
    rank.yaml
    rerank.yaml
  dataset/
  artifacts/
  redbookrec/
    __init__.py
    config.py
    data.py
    features.py
    metrics.py
    models.py
    recommend.py
    rerank.py
    search.py
    train_utils.py
  scripts/
    inspect_dataset.py
    prepare_data.py
    build_search_index.py
    train_twotower.py
    train_dcn_lite.py
    train_sim.py
    rerank_dpp.py
    evaluate.py
    recommend_user.py
```

Keep the first implementation small and runnable before adding deeper models.

## Implementation Plan

### Phase 1: Inspect and Prepare Data

- Confirm parquet schemas using `pandas` or `pyarrow`.
- Add the global config files before implementing stage scripts.
- Load config consistently in all scripts and save the merged config for each run.
- Build a prepared interaction table from `recommendation_train`.
- Build `user_id` and `note_id` mappings with `0` reserved for unknown/default IDs.
- Save unknown-ID statistics during preprocessing.
- Save prepared artifacts under `artifacts/`.
- Keep raw Qilin files unchanged.

Expected commands:

```bash
conda activate rs
python scripts/prepare_data.py
```

or:

```bash
conda run -n rs python scripts/prepare_data.py
```

### Phase 2: Build Baseline Recommender

- Implement search recall and Two-Tower recall first.
- For search recall, build a text index over note title, content, and taxonomy fields.
- Build user and note embeddings.
- Use approximate nearest neighbor search or batched dot-product retrieval for the first version.
- Return candidate note IDs with recall scores and recall source metadata.
- Route requests by query presence: keyword search uses search recall; empty-query feed uses Two-Tower recall.

Expected command:

```bash
conda run -n rs python scripts/build_search_index.py
conda run -n rs python scripts/train_twotower.py
```

### Phase 3: Pre-rank and Fine-rank

- Train DCN-lite on exposed candidates with click or engagement labels.
- Train SIM with reconstructed long user behavior sequences.
- In SIM, combine `lastN` recent behaviors with candidate-aware `topK` long-history retrieval.
- Use DCN-lite to reduce candidates, then SIM to produce high-quality ranking scores.

Expected command:

```bash
conda run -n rs python scripts/train_dcn_lite.py
conda run -n rs python scripts/train_sim.py --sim-last-n 20 --sim-top-k 20 --sim-max-history 200
```

### Phase 4: DPP Re-rank and Demo App

- Use DPP to select the final list from top SIM candidates.
- Combine predicted relevance with item similarity based on taxonomy and text or embedding features.
- Use Streamlit for the first demo unless the project later needs a FastAPI service.
- Show user history and generated recommendations on one page.
- Keep each recommendation card focused on the note itself, score, and reason.

Expected command:

```bash
conda run -n rs streamlit run app.py
```

## Later Extensions

- Add Popular, ItemCF, and category recall as fallback or auxiliary feed recall channels.
- Add dense semantic search recall for keyword requests.
- Add ANN indexing with Faiss for faster Two-Tower retrieval.
- Add multi-task learning for click, like, collect, comment, and share.
- Add image features after downloading Qilin image data.
- Add multimodal note embeddings to Two-Tower, SIM, and DPP similarity.
- Add DQA/RAG-based recommendation explanation once the core recommender is stable.

## Development Rules

- Keep raw files in `dataset/` read-only.
- Write generated data, indexes, and models to `artifacts/`.
- Prefer small, testable modules over one large script.
- Add dependencies to `requirements.txt` when they become necessary.
- Use `rs` for all Python, training, evaluation, and app commands.
- Do not assume packages are available in `base`.
- Local tests only need to prove the pipeline can run; they do not need to consume the full Qilin dataset.
- Keep all model code device-aware so the same scripts can run on CPU locally and on GPU servers later.
- All scripts must read shared settings from `configs/` and allow explicit CLI overrides.

## Final README Requirement

Do not create or polish the root `README.md` until the implementation is stable enough to document accurately. The final README must be written in Chinese and should explain:

- project background and Qilin dataset usage,
- Search Recall + Two-Tower, DCN-lite, SIM with `lastN` and `topK`, and DPP,
- local debug commands and GPU full-training commands,
- training, evaluation, prediction, and demo commands,
- ID mapping, default IDs, and cold-start behavior,
- expected project structure and generated artifact locations.

The final README should use general environment wording and must not mention the user's private conda environment name.
