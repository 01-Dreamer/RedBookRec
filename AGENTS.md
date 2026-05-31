# AGENTS.md

本文件用于指导 Codex 在 `RedBookRec` 项目中实现完整推荐系统。Codex 需要按照本文要求创建项目目录、实现代码、生成配置文件，并最终生成中文 `README.md`、`requirements.txt` 和 `.gitignore`。

当前项目根目录：

```bash
/home/failedman/RedBookRec
```

当前已有内容：

```text
RedBookRec/
├── dataset/
├── .git/
└── .gitignore
```

注意：用户只需要本文件作为给 Codex 的总指令文件。`README.md`、`requirements.txt`、`.gitignore` 不需要现在手写给用户，由 Codex 在实现项目时根据本文要求生成。其中 `README.md` 必须使用中文。

---

## 1. 项目目标

基于 Qilin 小红书数据集，实现一个四阶段推荐系统：

```text
召回：Search Recall + Dual-Tower Recall
  → 粗排：DCN
  → 精排：SIM
  → 重排：DPP
  → TopK 推荐列表
```

当前版本优先实现 **text-only** 版本，不使用图片和视频原始数据。Qilin 的图片和视频数据体量较大，首版只使用文本、ID、类别特征、统计特征和用户特征，确保完整流程可以在当前机器上跑通。

---

## 2. 运行环境与硬件限制

### 2.1 Conda 环境

必须使用用户已有 conda 环境：

```bash
conda activate rs
```

不要新建 conda 环境。所有脚本默认在项目根目录执行：

```bash
cd ~/RedBookRec
conda activate rs
```

若缺依赖，Codex 应创建或更新 `requirements.txt`，并提示用户执行：

```bash
pip install -r requirements.txt
```

### 2.2 当前硬件

用户当前开发机器为：

```text
CPU: 16 cores
RAM: 24 GB
GPU: 无独立 GPU，仅集成显卡
```

因此实现代码时必须遵守：

1. 首要目标是 **训练、推理、评估流程能跑通**，不是完整跑完全量大规模训练。
2. 默认配置必须适合 CPU 小样本 smoke test。
3. 不要默认要求独立 GPU。
4. 不要默认端到端训练 BERT、Qwen、VLM 等大模型。
5. 不读取图片和视频原始文件。
6. FAISS 默认使用 `faiss-cpu`。
7. 大文件必须分列读取、分批处理。
8. 所有阶段都必须支持小样本参数，便于 CPU 验证。

所有耗时脚本都要支持类似参数：

```text
--smoke-test
--max-notes
--max-requests
--max-train-samples
--max-eval-samples
```

默认训练配置建议：

```yaml
train:
  device: auto
  batch_size: 64
  num_workers: 0
```

代码中统一使用：

```python
def get_device(device: str):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
```

将来用户换到 GPU 机器时，只需修改配置：

```yaml
train:
  device: cuda
```

不要在代码中写死 CPU 或 CUDA。

---

## 3. 数据集概览

项目数据位于：

```text
dataset/
├── dqa/
├── notes/
├── recommendation_train/
├── recommendation_test/
├── search_train/
├── search_test/
└── user_feat/
```

### 3.1 表级概览

| 数据集 | 行数 | 主要作用 |
|---|---:|---|
| `dqa` | 6,972 | DQA/RAG 相关，推荐主流程首版不用 |
| `notes` | 1,983,938 | 全量 note 物料库，召回、排序和重排都要用 |
| `recommendation_train` | 83,437 | 推荐训练请求 |
| `recommendation_test` | 11,115 | 推荐测试请求 |
| `search_train` | 44,024 | 搜索训练请求，可用于 Search Recall |
| `search_test` | 6,192 | 搜索测试请求，可用于 Search Recall 评估 |
| `user_feat` | 15,482 | 用户画像和 dense features |

### 3.2 notes 字段

`notes` 字段：

```text
note_title, note_content, note_type,
video_duration, video_height, video_width,
image_num, content_length, commercial_flag,
taxonomy1_id, taxonomy2_id, taxonomy3_id,
imp_num, imp_rec_num, imp_search_num,
click_num, click_rec_num, click_search_num,
like_num, collect_num, comment_num, share_num,
screenshot_num, hide_num,
rec_like_num, rec_collect_num, rec_comment_num, rec_share_num, rec_follow_num,
search_like_num, search_collect_num, search_comment_num, search_share_num, search_follow_num,
accum_like_num, accum_collect_num, accum_comment_num,
view_time, rec_view_time, search_view_time, valid_view_times, full_view_times,
note_idx, image_path
```

首版 text-only 使用：

```text
note_idx
note_title
note_content
note_type
content_length
commercial_flag
taxonomy1_id
taxonomy2_id
taxonomy3_id
```

首版不读取 `image_path` 指向的文件，但可以保留 `image_num`、`note_type`、`video_duration` 这类轻量结构字段作为可选特征。

### 3.3 recommendation_train / recommendation_test 字段

`recommendation_train`：

```text
recent_clicked_note_idxs, request_idx, session_idx, user_idx, query, rec_result_details_with_idx
```

`recommendation_test`：

```text
recent_clicked_note_idxs, request_idx, session_idx, user_idx, rec_results, query, rec_result_details_with_idx
```

其中：

```text
recent_clicked_note_idxs: 用户请求前最近点击 note 列表，通常最多 20 个
rec_result_details_with_idx: 推荐候选列表
```

`rec_result_details_with_idx` 每项包含：

```text
click, collect, comment, like, note_idx, page_time, position, timestamp, share
```

### 3.4 search_train / search_test 字段

`search_train`：

```text
query, query_from_type, recent_clicked_note_idxs, search_idx, session_idx, user_idx,
dpr_results, search_result_details_with_idx
```

`search_test`：

```text
query, query_from_type, recent_clicked_note_idxs, search_idx, session_idx, user_idx,
bm25_results, dpr_results, search_results, search_result_details_with_idx
```

其中：

```text
bm25_results / dpr_results = [[note_idx, score], ...]
```

### 3.5 user_feat 字段

```text
gender, platform, age, fans_num, follows_num,
dense_feat1 ... dense_feat40,
location, user_idx
```

首版 User Tower 可以先只用：

```text
user_idx
recent_clicked_note_idxs
```

第二版再加入：

```text
gender, platform, age, fans_num, follows_num, dense_feat1...dense_feat40
```

---

## 4. 嵌套字段解析要求

这些字段可能是 list、numpy array、字符串化 JSON 或字符串化 Python literal：

```text
recent_clicked_note_idxs
rec_result_details_with_idx
search_result_details_with_idx
bm25_results
dpr_results
rec_results
search_results
```

必须实现鲁棒解析函数，例如：

```python
def parse_nested(value):
    ...
```

解析规则：

1. 如果已经是 list / tuple / numpy array，转成 list。
2. 如果是空值，返回空 list。
3. 如果是字符串，先尝试 `json.loads`。
4. 失败后尝试 `ast.literal_eval`。
5. 仍失败则返回空 list，并记录 warning。

---

## 5. 目标目录结构

Codex 需要创建并维护如下目录：

```text
RedBookRec/
├── README.md
├── AGENTS.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── recall.yaml
│   ├── dcn.yaml
│   ├── sim.yaml
│   └── dpp.yaml
│
├── dataset/
│   ├── notes/
│   ├── recommendation_train/
│   ├── recommendation_test/
│   ├── search_train/
│   ├── search_test/
│   ├── user_feat/
│   └── dqa/
│
├── data_cache/
│   ├── notes/
│   ├── recall/
│   ├── search_recall/
│   ├── hybrid_recall/
│   ├── prerank/
│   ├── rank/
│   └── rerank/
│
├── checkpoints/
│   ├── recall/
│   ├── prerank/
│   └── rank/
│
├── outputs/
│   ├── recall/
│   ├── search_recall/
│   ├── hybrid_recall/
│   ├── prerank/
│   ├── rank/
│   └── rerank/
│
├── scripts/
│   ├── analyze_notes.py
│   ├── analyze_recommendation.py
│   ├── prepare_notes.py
│   ├── build_recall_samples.py
│   ├── train_recall.py
│   ├── infer_recall.py
│   ├── run_search_recall.py
│   ├── merge_recall.py
│   ├── train_prerank.py
│   ├── infer_prerank.py
│   ├── train_rank.py
│   ├── infer_rank.py
│   ├── run_dpp.py
│   └── evaluate.py
│
└── src/
    └── redbookrec/
        ├── __init__.py
        ├── data/
        │   ├── __init__.py
        │   ├── load_qilin.py
        │   ├── preprocess_notes.py
        │   ├── preprocess_rec.py
        │   ├── preprocess_search.py
        │   ├── id_mapping.py
        │   └── sample_builder.py
        ├── features/
        │   ├── __init__.py
        │   ├── text.py
        │   ├── categorical.py
        │   ├── sequence.py
        │   └── numeric.py
        ├── recall/
        │   ├── __init__.py
        │   ├── dataset.py
        │   ├── model.py
        │   ├── loss.py
        │   ├── trainer.py
        │   ├── faiss_index.py
        │   └── inference.py
        ├── search_recall/
        │   ├── __init__.py
        │   ├── bm25.py
        │   ├── dpr.py
        │   ├── merger.py
        │   └── inference.py
        ├── prerank/
        │   ├── __init__.py
        │   ├── dataset.py
        │   ├── dcn.py
        │   ├── trainer.py
        │   └── inference.py
        ├── rank/
        │   ├── __init__.py
        │   ├── dataset.py
        │   ├── sim.py
        │   ├── trainer.py
        │   └── inference.py
        ├── rerank/
        │   ├── __init__.py
        │   ├── dpp.py
        │   └── inference.py
        ├── evaluation/
        │   ├── __init__.py
        │   ├── metrics.py
        │   ├── recall_eval.py
        │   ├── ranking_eval.py
        │   └── diversity_eval.py
        └── utils/
            ├── __init__.py
            ├── config.py
            ├── logger.py
            ├── seed.py
            └── io.py
```

不要创建 `notebooks/` 目录。

---

## 6. Codex 需要生成的基础文件

Codex 需要自行生成以下文件。

### 6.1 README.md

`README.md` 必须使用中文，内容包括：

1. 项目简介。
2. 数据集说明。
3. 四阶段推荐流程。
4. 环境安装，必须写明使用 `conda activate rs`。
5. 目录结构。
6. 数据分析、预处理、训练、推理、评估命令。
7. 四个阶段的测试脚本命令。
8. 输出文件说明。
9. 当前版本限制：text-only，不使用图片/视频。
10. 后续计划：多模态、user_feat 增强、hard negative、完整 DPP MAP。

### 6.2 requirements.txt

Codex 需要根据实际代码生成 `requirements.txt`。首版建议包含：

```text
numpy
pandas
pyarrow
scikit-learn
tqdm
pyyaml
torch
faiss-cpu
rank-bm25
```

若使用其他包，必须同步更新。

### 6.3 .gitignore

Codex 需要生成或更新 `.gitignore`，至少包含：

```gitignore
dataset/
data_cache/
checkpoints/
outputs/

*.pt
*.pth
*.ckpt
*.npy
*.npz
*.parquet
*.pkl
*.index

__pycache__/
*.pyc
.env
.vscode/
.idea/
.DS_Store
```

---

## 7. 配置文件要求

所有脚本都必须支持：

```bash
--config configs/xxx.yaml
```

### 7.1 configs/recall.yaml

首版建议内容：

```yaml
seed: 2025

data:
  dataset_dir: dataset
  cache_dir: data_cache
  note_text_path: data_cache/notes/note_text.parquet
  note_id_map_path: data_cache/notes/note_id_map.json
  note_text_emb_path: data_cache/notes/note_text_emb.npy
  train_samples_path: data_cache/recall/recall_train_samples.parquet
  test_samples_path: data_cache/recall/recall_test_samples.parquet

model:
  embed_dim: 128
  text_emb_dim: 384
  max_history_len: 20
  history_encoder: mean
  dropout: 0.1
  temperature: 0.05
  use_text_emb: false
  use_taxonomy: true
  use_note_type: true

train:
  device: auto
  batch_size: 64
  epochs: 1
  lr: 0.0001
  weight_decay: 0.00001
  num_workers: 0
  max_train_samples: 20000
  smoke_test: true

infer:
  top_k: 1000
  max_notes: 50000
  max_requests: 1000
  note_emb_path: data_cache/recall/note_emb.npy
  faiss_index_path: data_cache/recall/faiss.index
  faiss_note_ids_path: data_cache/recall/faiss_note_ids.npy
  dual_output_path: outputs/recall/test_top1000.parquet
  search_output_path: outputs/search_recall/test_top1000.parquet
  hybrid_output_path: outputs/hybrid_recall/test_top1000.parquet

hybrid_recall:
  w_dual: 0.7
  w_search: 0.3
```

注意：默认配置是 CPU smoke test 配置。完整实验时用户可以将 `smoke_test` 设为 false，并移除 `max_*` 限制。

### 7.2 其他配置

还需要生成：

```text
configs/dcn.yaml
configs/sim.yaml
configs/dpp.yaml
```

这些配置首版可以是最小可运行配置，但必须支持 CPU smoke test 和未来 GPU。

---

## 8. 阶段一：召回

召回包括：

```text
Search Recall
Dual-Tower Recall
Hybrid Recall
```

### 8.1 Dual-Tower Recall

实现文件：

```text
src/redbookrec/recall/
```

必须实现：

```python
class UserTower(nn.Module): ...
class NoteTower(nn.Module): ...
class DualTowerRecall(nn.Module): ...
```

User Tower 首版输入：

```text
user_idx
recent_clicked_note_idxs
```

结构：

```text
user embedding
history note embedding mean pooling
concat
MLP
L2 normalize
```

Note Tower 首版输入：

```text
note_idx
note_type
taxonomy1_id
taxonomy2_id
taxonomy3_id
note_text_emb 可选
```

结构：

```text
note embedding
type embedding
taxonomy embedding
text projection 可选
concat
MLP
L2 normalize
```

训练损失：

```text
In-batch softmax / InfoNCE
```

```python
logits = user_emb @ note_emb.T / temperature
labels = torch.arange(batch_size)
loss = CrossEntropyLoss(logits, labels)
```

训练脚本：

```bash
python scripts/train_recall.py --config configs/recall.yaml
```

推理脚本：

```bash
python scripts/infer_recall.py --config configs/recall.yaml
```

输出：

```text
outputs/recall/test_top1000.parquet
```

字段：

```text
request_idx
user_idx
note_idx
recall_score
recall_rank
label_click
source
```

### 8.2 Search Recall

实现文件：

```text
src/redbookrec/search_recall/
```

脚本：

```bash
python scripts/run_search_recall.py --config configs/recall.yaml
```

首版策略：

1. 对 `notes.note_text` 建 BM25 索引，或使用已有 `bm25_results/dpr_results` 做可选 fallback。
2. recommendation request 的 `query` 非空时使用 query。
3. query 为空时，用 `recent_clicked_note_idxs` 对应 note 标题拼成 pseudo-query。
4. 输出 Top1000。

输出：

```text
outputs/search_recall/test_top1000.parquet
```

字段：

```text
request_idx
user_idx
note_idx
search_score
search_rank
label_click
source
```

### 8.3 Hybrid Recall

脚本：

```bash
python scripts/merge_recall.py --config configs/recall.yaml
```

输入：

```text
outputs/recall/test_top1000.parquet
outputs/search_recall/test_top1000.parquet
```

输出：

```text
outputs/hybrid_recall/test_top1000.parquet
```

合并规则：

```text
hybrid_score = w_dual * norm(dual_score) + w_search * norm(search_score)
```

按 `request_idx, note_idx` 去重，每个 request 保留 Top1000。

---

## 9. 阶段二：粗排 DCN

配置：

```text
configs/dcn.yaml
```

脚本：

```bash
python scripts/train_prerank.py --config configs/dcn.yaml
python scripts/infer_prerank.py --config configs/dcn.yaml
```

输入：

```text
outputs/hybrid_recall/test_top1000.parquet
data_cache/notes/note_text.parquet
dataset/user_feat/*.parquet
```

如果首版没有 hybrid recall 的 train candidates，可用 `recommendation_train` 原始曝光候选构造训练样本。

模型：

```text
DCN = embedding layer + 1~2 cross layers + shallow MLP
```

特征：

```text
user_idx
note_idx
note_type
taxonomy1/2/3
content_length
commercial_flag
dual_score
search_score
hybrid_score
hybrid_rank
user_feat 可选
```

标签：

```text
click
```

输出：

```text
outputs/prerank/test_top200.parquet
```

字段：

```text
request_idx
user_idx
note_idx
hybrid_score
dcn_score
dcn_rank
label_click
```

每个 request 保留 Top200。

---

## 10. 阶段三：精排 SIM

配置：

```text
configs/sim.yaml
```

脚本：

```bash
python scripts/train_rank.py --config configs/sim.yaml
python scripts/infer_rank.py --config configs/sim.yaml
```

输入：

```text
outputs/prerank/test_top200.parquet
recent_clicked_note_idxs
note embedding
```

首版实现简化 SIM：

1. 候选 note 作为 target。
2. 与用户历史 note embedding 计算相似度。
3. GSU：取 TopK 相关历史行为。
4. ESU：对 TopK 历史行为做 target attention。
5. MLP 输出 `sim_score`。

输出：

```text
outputs/rank/test_top50.parquet
```

字段：

```text
request_idx
user_idx
note_idx
dcn_score
sim_score
sim_rank
label_click
```

每个 request 保留 Top50。

---

## 11. 阶段四：重排 DPP

配置：

```text
configs/dpp.yaml
```

脚本：

```bash
python scripts/run_dpp.py --config configs/dpp.yaml
```

输入：

```text
outputs/rank/test_top50.parquet
note embeddings
```

首版实现 greedy DPP-inspired rerank：

```text
score = relevance - lambda_diversity * max_similarity_to_selected
```

其中：

```text
relevance = sim_score
similarity = cosine(note_emb_i, note_emb_j)
```

输出：

```text
outputs/rerank/test_top10.parquet
```

字段：

```text
request_idx
user_idx
note_idx
sim_score
dpp_score
final_rank
label_click
```

每个 request 输出 Top10。

---

## 12. 评估要求

统一脚本：

```bash
python scripts/evaluate.py --config configs/recall.yaml --stage recall
python scripts/evaluate.py --config configs/dcn.yaml --stage prerank
python scripts/evaluate.py --config configs/sim.yaml --stage rank
python scripts/evaluate.py --config configs/dpp.yaml --stage rerank
```

### 12.1 召回指标

```text
Recall@50
Recall@100
Recall@500
Recall@1000
MRR@100
NDCG@100
```

### 12.2 排序指标

```text
AUC
LogLoss
MRR@10
NDCG@10
MAP@10
Recall@10
```

### 12.3 多样性指标

```text
ILD@10
taxonomy coverage@10
note_type diversity@10
```

---

## 13. 数据预处理脚本要求

### 13.1 analyze_notes.py

命令：

```bash
python scripts/analyze_notes.py --config configs/recall.yaml
```

功能：

- 打印 notes 行数、字段、缺失率。
- 打印 note_type 分布。
- 打印 taxonomy 缺失情况。
- 打印文本长度统计。
- 支持 `--max-notes`。

### 13.2 analyze_recommendation.py

命令：

```bash
python scripts/analyze_recommendation.py --config configs/recall.yaml
```

功能：

- 打印 request 数。
- 打印每个 request 候选数量。
- 打印点击率。
- 打印正负样本比例。
- 检查候选 note 是否能在 notes 中找到。
- 支持 `--max-requests`。

### 13.3 prepare_notes.py

命令：

```bash
python scripts/prepare_notes.py --config configs/recall.yaml
```

输入：

```text
dataset/notes/*.parquet
```

输出：

```text
data_cache/notes/note_text.parquet
data_cache/notes/note_id_map.json
data_cache/notes/note_text_emb.npy    # 可选
```

必须处理：

- `note_idx` 转 int。
- `note_title`、`note_content` 缺失填空。
- taxonomy 中 `"nan"`、空值转 `"UNK"`。
- 构造：

```text
note_text = note_title + " [SEP] " + note_content
```

- 生成 note ID 映射：

```text
raw note_idx -> model note_id，从 1 开始，0 留给 padding
```

### 13.4 build_recall_samples.py

命令：

```bash
python scripts/build_recall_samples.py --config configs/recall.yaml
```

输出：

```text
data_cache/recall/recall_train_samples.parquet
data_cache/recall/recall_test_samples.parquet
```

每条样本字段：

```text
request_idx
session_idx
user_idx
recent_clicked_note_idxs
pos_note_idx
label
```

正样本：

```text
rec_result_details_with_idx 中 click = 1 的 note
```

首版双塔召回使用 in-batch negative，所以样本文件只保存正样本即可。

---

## 14. 最小验收标准

Codex 首个可验收版本必须能在 CPU smoke test 下执行：

```bash
conda activate rs
cd ~/RedBookRec

python scripts/analyze_notes.py --config configs/recall.yaml --max-notes 50000
python scripts/analyze_recommendation.py --config configs/recall.yaml --max-requests 5000
python scripts/prepare_notes.py --config configs/recall.yaml --max-notes 50000
python scripts/build_recall_samples.py --config configs/recall.yaml --max-requests 5000
python scripts/train_recall.py --config configs/recall.yaml --smoke-test
python scripts/infer_recall.py --config configs/recall.yaml --max-notes 50000 --max-requests 1000
python scripts/evaluate.py --config configs/recall.yaml --stage recall
```

必须生成：

```text
data_cache/notes/note_text.parquet
data_cache/notes/note_id_map.json
data_cache/recall/recall_train_samples.parquet
data_cache/recall/recall_test_samples.parquet
checkpoints/recall/best.pt
outputs/recall/test_top1000.parquet
outputs/recall/recall_metrics.json
```

完整数据训练不是首版验收要求，但代码必须保留完整数据运行能力。

---

## 15. 推荐开发顺序

Codex 请按以下顺序实现：

1. 创建目录结构。
2. 生成 `README.md`，必须中文。
3. 生成 `requirements.txt`。
4. 更新 `.gitignore`。
5. 创建配置文件：
   - `configs/recall.yaml`
   - `configs/dcn.yaml`
   - `configs/sim.yaml`
   - `configs/dpp.yaml`
6. 实现公共工具：
   - `utils/config.py`
   - `utils/logger.py`
   - `utils/seed.py`
   - `utils/io.py`
7. 实现数据读取与解析：
   - `data/load_qilin.py`
   - `data/preprocess_notes.py`
   - `data/preprocess_rec.py`
   - `data/preprocess_search.py`
   - `data/sample_builder.py`
8. 实现分析脚本：
   - `scripts/analyze_notes.py`
   - `scripts/analyze_recommendation.py`
9. 实现 notes 预处理：
   - `scripts/prepare_notes.py`
10. 实现召回样本构造：
   - `scripts/build_recall_samples.py`
11. 实现双塔召回：
   - `recall/dataset.py`
   - `recall/model.py`
   - `recall/loss.py`
   - `recall/trainer.py`
   - `recall/faiss_index.py`
   - `recall/inference.py`
12. 实现召回训练、推理、评估脚本。
13. 实现 Search Recall 与 Hybrid Recall。
14. 实现 DCN 粗排。
15. 实现 SIM 精排。
16. 实现 DPP 重排。
17. 最后补充和完善中文 `README.md`。

---

## 16. 重要提醒

- 首版不读取图片和视频原始文件。
- 首版不要端到端训练大模型。
- 首版要保证 CPU 小样本可跑通。
- 完整训练可以留给用户后续在更强机器上运行。
- 所有阶段都要保留 GPU 接口。
- 所有阶段都要有脚本可测试。
- `README.md` 必须中文。
- Codex 生成代码后，要优先运行 smoke test 验证。
