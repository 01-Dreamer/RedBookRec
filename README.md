# RedBookRec

RedBookRec 是一个基于 Qilin 数据集的小红书风格内容推荐系统项目。项目目标是构建一条完整的离线推荐链路，覆盖数据预处理、召回、粗排、精排、重排、评测和单用户预测。

本项目使用公开 Qilin 数据集实现“小红书风格”的内容流推荐系统，不代表也不声称复现小红书真实线上推荐系统。

## 系统架构

RedBookRec 当前采用四阶段推荐架构：

| 阶段 | 算法 | 作用 |
| --- | --- | --- |
| 召回 | Search Recall + Two-Tower | 有关键词时走 BM25 + TF-IDF 搜索召回；无关键词时走 Two-Tower 个性化召回 |
| 粗排 | DCN-lite | 使用轻量特征交叉模型快速过滤弱相关候选 |
| 精排 | SIM | 使用 `lastN` 近期行为和 `topK` 长历史目标相关行为建模用户兴趣 |
| 重排 | DPP | 在高分候选中选择更高质量且更多样的最终列表 |

整体流程：

```text
Qilin dataset
  -> inspect_dataset
  -> prepare_data
  -> build_search_index / train_twotower
  -> train_dcn_lite
  -> train_sim
  -> rerank_dpp
  -> evaluate
  -> recommend_user / app.py
```

## 数据集目录

默认数据集目录为 `dataset/`。

主要使用：

| 路径 | 用途 |
| --- | --- |
| `dataset/recommendation_train/` | 推荐训练日志 |
| `dataset/recommendation_test/` | 推荐测试日志 |
| `dataset/notes/` | 笔记内容库 |
| `dataset/user_feat/` | 用户画像和 dense 用户特征 |
| `dataset/search_train/` | 搜索训练数据，可用于搜索召回优化 |
| `dataset/search_test/` | 搜索测试数据，可用于搜索召回评测 |
| `dataset/dqa/` | 后续可用于推荐解释或 RAG |

原始数据保持只读。预处理结果、索引、模型和评测结果默认写入 `artifacts/`。

## 环境安装

建议使用 Python 虚拟环境或 conda 环境，环境名称不限。

```bash
pip install -r requirements.txt
```

如果暂时不需要 Streamlit，也可以先只跑命令行链路。当前 `app.py` 在未安装 Streamlit 时会提示改用 `scripts/recommend_user.py`。

## 配置系统

项目使用统一配置目录：

```text
configs/
  base.yaml
  debug.yaml
  full.yaml
  recall.yaml
  rank.yaml
  rerank.yaml
```

配置覆盖顺序：

```text
base.yaml
  -> 阶段配置，例如 recall.yaml / rank.yaml
  -> debug.yaml 或 full.yaml
  -> 命令行参数
```

本地 16 核 CPU、24 GB 内存环境只需要跑通小样本 smoke test。全量训练建议迁移到 GPU 服务器后再执行。

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--debug` | 小样本调试模式 |
| `--full` | 全量模式 |
| `--device cpu|cuda|auto` | 指定运行设备 |
| `--max-users` | 限制用户数 |
| `--max-notes` | 限制笔记数 |
| `--max-interactions` | 限制交互样本数 |
| `--batch-size` | batch 大小 |
| `--num-workers` | DataLoader worker 数 |
| `--epochs` | 训练轮数 |
| `--mixed-precision` | GPU 混合精度 |

## 运行命令

以下命令均在项目根目录执行。

### 1. 检查数据 Schema

```bash
python scripts/inspect_dataset.py --debug
```

输出会保存到：

```text
artifacts/processed/schema_summary.json
```

### 2. 数据预处理

```bash
python scripts/prepare_data.py --debug --max-users 500 --max-notes 5000 --max-interactions 5000
```

预处理会完成：

- 展开 `rec_result_details_with_idx`
- 构建曝光、点击、互动样本
- 生成 `click_label`、`engage_label`、`relevance`
- 构建用户 ID 和笔记 ID 映射
- 预留默认 ID：未知用户和未知笔记均映射到 `0`
- 构建用户历史行为序列
- 拼接笔记特征和用户特征

主要输出：

```text
artifacts/processed/interactions.parquet
artifacts/processed/training_features.parquet
artifacts/processed/notes.parquet
artifacts/processed/users.parquet
artifacts/processed/user_history.parquet
artifacts/processed/mappings/
```

### 3. 构建搜索召回索引

```bash
python scripts/build_search_index.py --debug --max-notes 5000
```

当前搜索召回使用 BM25 + TF-IDF 混合检索：

```text
query -> BM25 + TF-IDF -> note title/content/taxonomy -> topK notes
```

输出：

```text
artifacts/indexes/search_index.joblib
artifacts/indexes/search_tfidf.joblib
```

### 4. 训练 Two-Tower 召回模型

```bash
python scripts/train_twotower.py --debug --device cpu --epochs 1 --batch-size 128 --num-workers 0 --max-users 20 --max-interactions 500
```

输出：

```text
artifacts/checkpoints/twotower.pt
artifacts/indexes/twotower_candidates.joblib
artifacts/metrics/twotower_train.json
```

### 5. 训练 DCN-lite 粗排模型

```bash
python scripts/train_dcn_lite.py --debug --device cpu --epochs 1 --batch-size 128 --num-workers 0 --max-interactions 500
```

输出：

```text
artifacts/checkpoints/dcn_lite.pt
artifacts/metrics/dcn_train.json
```

### 6. 训练 SIM 精排模型

```bash
python scripts/train_sim.py --debug --device cpu --epochs 1 --batch-size 128 --num-workers 0 --max-interactions 500 --sim-last-n 5 --sim-top-k 5 --sim-max-history 50
```

SIM 当前设计：

- `lastN`：用户最近 N 条点击或互动行为，表达短期兴趣
- `topK`：从用户长历史中取与候选笔记最相关的 K 条行为，表达长期目标相关兴趣
- 新用户或历史不足时使用 `note_id = 0` padding

输出：

```text
artifacts/checkpoints/sim.pt
artifacts/metrics/sim_train.json
```

### 7. DPP 重排

```bash
python scripts/rerank_dpp.py --debug --top-k 10
```

DPP 重排会结合：

- 历史已看笔记过滤
- 重复 `note_id` / `raw_note_idx` 过滤
- 标题和正文近似重复过滤
- taxonomy 相似度多样性约束

输出：

```text
artifacts/processed/sample_dpp_rerank.parquet
```

### 8. 离线评测

```bash
python scripts/evaluate.py --debug --top-k 10 --max-interactions 1000
```

输出：

```text
artifacts/metrics/eval_summary.json
```

当前评测包含：

- `HitRate@K`
- `MRR@K`
- `NDCG@K`
- `Recall@K`

### 9. 单用户推荐

无关键词 Feed 推荐：

```bash
python scripts/recommend_user.py --debug --random-user --top-k 5
```

指定用户：

```bash
python scripts/recommend_user.py --debug --user-id 35 --top-k 5
```

带关键词搜索推荐：

```bash
python scripts/recommend_user.py --debug --user-id 0 --query "春季穿搭" --top-k 5
```

### 10. Streamlit Demo

安装 Streamlit 后可运行：

```bash
streamlit run app.py
```

如果没有安装 Streamlit，请先使用命令行推荐脚本。

## GPU 全量训练示例

迁移到 GPU 服务器后，可以使用：

```bash
python scripts/prepare_data.py --full
python scripts/build_search_index.py --full
python scripts/train_twotower.py --full --device cuda --epochs 10 --batch-size 4096 --mixed-precision
python scripts/train_dcn_lite.py --full --device cuda --epochs 10 --batch-size 4096 --mixed-precision
python scripts/train_sim.py --full --device cuda --epochs 5 --batch-size 1024 --mixed-precision --sim-last-n 50 --sim-top-k 100 --sim-max-history 1000
python scripts/evaluate.py --full --top-k 100
```

## ID 映射与冷启动

所有 ID 映射都预留默认值：

| 类型 | 默认 ID | 说明 |
| --- | --- | --- |
| 用户 | `0` | 未知用户、新用户、缺失用户 |
| 笔记 | `0` | 未知笔记、新笔记、缺失笔记 |
| 类目 | `0` | 未知类目、缺失类目 |
| query/token | `0` | 空 query 或未知 token |

真实数据 ID 从 `1` 开始映射。模型 embedding 中的 `0` 作为 padding / unknown 使用。

冷启动策略：

- 新用户有 query：走搜索召回
- 新用户无 query：走热门或其他 fallback 候选
- 新笔记：先用默认 note ID，并结合文本、类目和统计特征参与排序
- 历史不足：使用 `note_id = 0` padding

## 项目结构

```text
RedBookRec/
  app.py
  requirements.txt
  AGENTS.md
  README.md
  configs/
  dataset/
  artifacts/
  redbookrec/
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

## 当前状态

当前版本已经完成一条 debug 小样本链路：

```text
schema inspect
  -> data prepare
  -> BM25 + TF-IDF search recall
  -> Two-Tower recall
  -> DCN-lite
  -> SIM
  -> DPP
  -> evaluate
  -> recommend_user
```

后续可以继续增强：

- Faiss ANN 检索
- 更完整的负采样策略
- 更强的 SIM topK 历史检索
- 更精细的多任务损失权重与目标校准
- 多模态图片特征
