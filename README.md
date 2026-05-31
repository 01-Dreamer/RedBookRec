# RedBookRec

RedBookRec 是基于 Qilin 小红书数据集的 text-only 推荐系统首版实现。项目目标是用轻量、CPU 可运行的方式打通完整推荐链路：

```text
Search Recall + Dual-Tower Recall
  -> Hybrid Recall
  -> DCN 粗排
  -> SIM 精排
  -> DPP 重排
  -> TopK 推荐列表
```

当前版本不读取图片和视频原始文件，只使用文本、ID、类别、统计字段和用户近期点击序列，优先保证数据分析、预处理、训练、推理、评估流程能在本机 smoke test 下跑通。

## 数据集

数据默认放在 `dataset/`：

| 子集 | 作用 |
|---|---|
| `notes` | note 物料库，提供标题、正文、类型、taxonomy 等 |
| `recommendation_train` | 推荐训练请求和曝光候选 |
| `recommendation_test` | 推荐测试请求和曝光候选 |
| `search_train` / `search_test` | 搜索相关请求，可用于搜索召回 |
| `user_feat` | 用户画像和 dense features，首版暂未深度使用 |
| `dqa` | DQA/RAG 数据，首版推荐主链路暂未使用 |

## 环境安装

使用已有 conda 环境：

```bash
cd ~/RedBookRec
conda activate rs
pip install -r requirements.txt
```

`faiss-cpu` 和 `rank-bm25` 写在依赖中；代码对它们做了降级处理，缺失时仍可用 sklearn / numpy 跑通 smoke test。

## 目录结构

```text
configs/                 # recall、dcn、sim、dpp 配置
scripts/                 # 数据分析、训练、推理、评估入口
src/redbookrec/          # 推荐系统源码
data_cache/              # 预处理数据和中间缓存
checkpoints/             # 模型检查点
outputs/                 # 各阶段推荐结果和指标
dataset/                 # Qilin 原始 parquet 数据
```

## 召回 Smoke Test

```bash
python scripts/analyze_notes.py --config configs/recall.yaml --max-notes 50000
python scripts/analyze_recommendation.py --config configs/recall.yaml --max-requests 5000
python scripts/prepare_notes.py --config configs/recall.yaml --max-notes 50000
python scripts/build_recall_samples.py --config configs/recall.yaml --max-requests 5000
python scripts/train_recall.py --config configs/recall.yaml --smoke-test
python scripts/infer_recall.py --config configs/recall.yaml --max-notes 50000 --max-requests 1000
python scripts/evaluate.py --config configs/recall.yaml --stage recall
```

主要产物：

```text
data_cache/notes/note_text.parquet
data_cache/notes/note_id_map.json
data_cache/recall/recall_train_samples.parquet
data_cache/recall/recall_test_samples.parquet
checkpoints/recall/best.pt
outputs/recall/test_top1000.parquet
outputs/recall/recall_metrics.json
```

## 四阶段命令

Search Recall 与 Hybrid Recall：

```bash
python scripts/run_search_recall.py --config configs/recall.yaml --max-notes 50000 --max-requests 1000
python scripts/merge_recall.py --config configs/recall.yaml
python scripts/evaluate.py --config configs/recall.yaml --stage hybrid_recall
```

DCN 粗排：

```bash
python scripts/train_prerank.py --config configs/dcn.yaml --smoke-test
python scripts/infer_prerank.py --config configs/dcn.yaml
python scripts/evaluate.py --config configs/dcn.yaml --stage prerank
```

SIM 精排：

```bash
python scripts/train_rank.py --config configs/sim.yaml --smoke-test
python scripts/infer_rank.py --config configs/sim.yaml
python scripts/evaluate.py --config configs/sim.yaml --stage rank
```

DPP 重排：

```bash
python scripts/run_dpp.py --config configs/dpp.yaml
python scripts/evaluate.py --config configs/dpp.yaml --stage rerank
```

## 输出说明

| 阶段 | 输出 |
|---|---|
| Dual-Tower Recall | `outputs/recall/test_top1000.parquet` |
| Search Recall | `outputs/search_recall/test_top1000.parquet` |
| Hybrid Recall | `outputs/hybrid_recall/test_top1000.parquet` |
| DCN | `outputs/prerank/test_top200.parquet` |
| SIM | `outputs/rank/test_top50.parquet` |
| DPP | `outputs/rerank/test_top10.parquet` |

## 当前限制

- 首版是 text-only，不读取 Qilin 图片和视频原始文件。
- 双塔召回使用 in-batch negative，默认配置面向 CPU smoke test。
- DCN、SIM、DPP 已提供可运行骨架和轻量策略，后续可替换为更强训练逻辑。
- 默认配置优先小样本验证，不代表全量最优效果。

## 后续计划

- 引入图片、多模态 note embedding。
- 更充分使用 `user_feat` 的画像和 dense features。
- 增加 hard negative、热门/ItemCF/category fallback 召回。
- 完善 SIM 的长序列行为建模。
- 实现完整 DPP MAP kernel 与更丰富的多样性评估。
