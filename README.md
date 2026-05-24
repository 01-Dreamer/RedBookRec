# RedBookRec

RedBookRec 是一个基于本地 Qilin 数据集的“小红书 / REDNote 风格”离线内容推荐系统项目。

这个项目的目标不是做一个小 demo，而是实现一套可阅读、可运行、适合学习和简历展示的推荐系统代码。它使用公开研究数据集 Qilin，模拟内容流推荐的常见离线流程；它不是小红书真实线上推荐系统的复现，也不声称还原小红书的真实推荐算法。

## 项目定位

本项目覆盖一条完整的离线推荐链路：

```text
本地 Qilin parquet 数据
  -> 数据结构检查
  -> 字段映射与样本预处理
  -> 用户画像与笔记画像
  -> 多路召回
  -> 候选合并
  -> 双塔召回模型
  -> DIN 风格排序模型
  -> 多目标打分
  -> 多样性重排
  -> 离线评估
  -> 用户推荐命令行示例
```

默认配置偏向本地 CPU 调试，适合在 16 核 CPU、24GB 内存这类机器上验证代码正确性。后续迁移到 GPU 服务器时，可以通过命令行参数或配置文件放大训练数据量、批大小和训练轮数。

## 数据说明

数据集已经在本地 `./dataset/` 目录下，不需要也不应该重新下载。

主要使用的数据子集：

- `notes`：笔记 / 内容池
- `user_feat`：用户特征表
- `recommendation_train`：推荐训练数据
- `recommendation_test`：推荐测试数据

暂时不作为第一阶段主链路的数据：

- `search_train`
- `search_test`
- `dqa`

代码不会盲目假设字段名。第一步会运行数据结构检查脚本，保存字段、类型、样例和嵌套字段摘要。之后解析器会把 Qilin 原始字段映射到项目内部统一字段，例如：

```text
user_idx                  -> user_id
note_idx                  -> note_id
recent_clicked_note_idxs  -> history_note_ids
rec_result_details_with_idx 中的 click/like/collect/comment/share/page_time
note_title                -> title
note_content              -> content
taxonomy1_id              -> category
```

缺失的可选字段会跳过；关键字段缺失时会给出明确报错，提示先运行结构检查脚本。

## 环境要求

使用已有 conda 环境 `rs`：

```bash
conda run -n rs python --version
```

安装依赖：

```bash
conda run -n rs pip install -r requirements.txt
```

不要使用 `base` 环境，也不要新建环境。所有脚本都按下面这种方式运行：

```bash
conda run -n rs python scripts/脚本名.py
```

## 本地 CPU 推荐运行方式

本地机器只建议做小规模验证，确认数据解析、训练、预测、评估链路没有问题。默认 `configs/base.yaml` 已经开启 debug 限制：

```yaml
debug:
  enabled: true
  max_users: 5000
  max_notes: 50000
  max_interactions: 200000
```

完整小规模流水线：

```bash
conda run -n rs python scripts/00_inspect_dataset.py
conda run -n rs python scripts/01_prepare_data.py
conda run -n rs python scripts/02_build_recall.py
conda run -n rs python scripts/03_train_twotower.py
conda run -n rs python scripts/04_train_ranker.py
conda run -n rs python scripts/05_evaluate.py --max-eval-users 100
conda run -n rs python scripts/06_recommend_user.py --random_user --top_k 20
```

如果只想快速检查排序模型训练代码：

```bash
conda run -n rs python scripts/04_train_ranker.py --epochs 1 --batch-size 512 --max-train-samples 50000
```

如果只想快速检查双塔训练代码：

```bash
conda run -n rs python scripts/03_train_twotower.py --epochs 1 --batch-size 512 --max-train-samples 50000
```

如果只想快速检查推荐输出：

```bash
conda run -n rs python scripts/06_recommend_user.py --random_user --top_k 10
```

## GPU 服务器运行方式

迁移到 GPU 服务器后，可以使用同一套脚本，通过 `--full` 关闭 debug 限制，通过 `--device cuda` 使用 GPU：

```bash
conda run -n rs python scripts/01_prepare_data.py --full
conda run -n rs python scripts/02_build_recall.py --full
conda run -n rs python scripts/03_train_twotower.py --full --device cuda --epochs 5 --batch-size 2048 --max-train-samples 2000000
conda run -n rs python scripts/04_train_ranker.py --full --device cuda --epochs 5 --batch-size 2048 --max-train-samples 2000000
conda run -n rs python scripts/05_evaluate.py --full --device cuda
conda run -n rs python scripts/06_recommend_user.py --random_user --top_k 20 --device cuda
```

也可以加载额外配置文件：

```bash
conda run -n rs python scripts/04_train_ranker.py --config configs/gpu.yaml --epochs 10 --batch-size 4096
```

常用可覆盖参数：

```text
--full                  关闭 debug 限制
--device cuda           使用 GPU
--device cpu            强制使用 CPU
--epochs                覆盖训练轮数
--batch-size            覆盖批大小
--max-train-samples     覆盖最大训练样本数
--max-users             覆盖 debug 用户数
--max-notes             覆盖 debug 笔记数
--max-interactions      覆盖 debug 交互数
--max-eval-users        覆盖评估用户数
--config                追加自定义 yaml 配置
```

## 脚本说明

### 1. 数据结构检查

```bash
conda run -n rs python scripts/00_inspect_dataset.py
```

输出：

- `outputs/schema_summary.json`
- `outputs/sample_rows/`

这个脚本会读取本地 parquet 文件，保存每个文件的路径、行数、列名、类型、样例行、嵌套/list 字段摘要。

### 2. 数据预处理

```bash
conda run -n rs python scripts/01_prepare_data.py
```

输出：

- `data_processed/samples/train_interactions.parquet`
- `data_processed/samples/test_interactions.parquet`
- `data_processed/features/note_features.parquet`
- `data_processed/features/user_features.parquet`
- `data_processed/features/user_profiles.parquet`
- `data_processed/mappings/id_mappings.json`

主要工作：

- 展开 `rec_result_details_with_idx`
- 生成点击、点赞、收藏、评论、分享、停留时长等标签
- 构建用户历史序列
- 构建 note_id/user_id 到内部连续 id 的映射
- 生成用户画像和笔记画像

### 3. 多路召回

```bash
conda run -n rs python scripts/02_build_recall.py
```

输出：

- `data_processed/recalls/classic_recall.parquet`
- `data_processed/recalls/merged_recall.parquet`
- `data_processed/recalls/popular_itemcf.pkl`

已实现召回：

- 热门召回
- ItemCF 召回
- TF-IDF 文本召回

文本召回默认在 debug 模式下限制笔记规模，避免本地 CPU 上过慢。迁移到服务器后可以通过配置调大。

### 4. 双塔召回模型训练

```bash
conda run -n rs python scripts/03_train_twotower.py
```

输出：

- `models/twotower/twotower.pt`
- `models/twotower/train_metrics.json`
- `data_processed/recalls/twotower_recall.parquet`

模型思路：

- 用户塔：用户 id embedding + 历史笔记 embedding 池化
- 物品塔：笔记 id embedding
- 训练目标：批内负样本 softmax
- 训练后会生成双塔向量召回结果，并合并进 `merged_recall.parquet`

### 5. DIN 风格排序模型训练

```bash
conda run -n rs python scripts/04_train_ranker.py
```

输出：

- `models/ranker/din_ranker.pt`
- `models/ranker/train_metrics.json`

模型思路：

- 对目标笔记和用户历史笔记做 embedding
- 使用 DIN 风格注意力，让目标笔记对历史序列做兴趣提取
- 拼接用户、目标、兴趣、位置、召回分等特征
- 多目标预测 click、like、collect、comment、share、page_time

如果某些标签不存在，训练逻辑会跳过不可用目标。

### 6. 离线评估

```bash
conda run -n rs python scripts/05_evaluate.py --max-eval-users 100
```

输出：

- `outputs/metrics/eval_summary.json`
- `outputs/metrics/eval_summary.csv`
- `data_processed/recalls/ranked_candidates.parquet`
- `data_processed/recalls/ranked_reranked.parquet`

评估指标：

- AUC
- LogLoss
- Recall@K
- HitRate@K
- NDCG@K
- MRR@K
- 覆盖度
- 平均推荐分

评估对象：

- 热门召回
- ItemCF 召回
- 双塔召回
- 合并召回
- DIN 排序
- DIN 排序 + 重排

本地建议加 `--max-eval-users`，避免评估候选过多。

### 7. 用户推荐命令

随机选一个用户：

```bash
conda run -n rs python scripts/06_recommend_user.py --random_user --top_k 20
```

指定用户：

```bash
conda run -n rs python scripts/06_recommend_user.py --user_id 6948 --top_k 20
```

输出内容：

- user_id
- 用户画像摘要
- 最近历史笔记
- Top-N 推荐笔记
- 标题和正文片段
- 推荐分
- 召回来源
- 重排原因

推荐结果也会保存到：

```text
outputs/recommendations/user_<id>_recommendations.parquet
```

## 可选 Streamlit 页面

如果安装了 Streamlit，可以启动简单页面：

```bash
conda run -n rs streamlit run app.py
```

页面包含：

- 用户选择
- 用户画像
- 推荐结果
- 推荐分
- 召回来源
- 重排说明

这是可选功能，核心链路仍然以脚本为主。

## 配置文件

主要配置：

- `configs/base.yaml`：默认路径、随机种子、debug 限制、评估限制
- `configs/recall.yaml`：召回参数
- `configs/twotower.yaml`：双塔训练参数
- `configs/ranker.yaml`：排序模型训练参数
- `configs/full.yaml`：关闭 debug 限制
- `configs/gpu.yaml`：GPU 服务器配置示例

自定义配置示例：

```bash
conda run -n rs python scripts/04_train_ranker.py --config configs/gpu.yaml --epochs 8
```

后加载的配置会覆盖先加载的配置。

## 重要产物目录

```text
data_processed/
  samples/      训练和测试样本
  features/     用户、笔记、画像特征
  mappings/     id 映射
  recalls/      召回、排序、重排候选

models/
  twotower/     双塔模型
  ranker/       DIN 排序模型

outputs/
  schema_summary.json
  sample_rows/
  metrics/
  recommendations/
  figures/
```

这些目录是运行产物，默认已加入 `.gitignore`，不要把大文件提交到仓库。

## 多目标打分

排序模型会按可用目标做加权融合。默认权重：

```text
click      0.45
collect    0.25
like       0.15
comment    0.05
share      0.05
page_time  0.05
```

如果部分目标不存在，会对剩余目标重新归一化权重。

## 重排规则

当前实现的是轻量规则重排：

- 过滤用户历史中已经看过的笔记
- 类目多样性控制
- 保留高分候选
- 缺少作者、类目、标签等字段时不会崩溃

后续可以继续加入作者去重、标签多样性、内容安全规则、新鲜度规则等。

## 本地已经验证过的轻量链路

当前代码已经在本地小规模跑通过：

```bash
conda run -n rs python scripts/00_inspect_dataset.py
conda run -n rs python scripts/01_prepare_data.py
conda run -n rs python scripts/02_build_recall.py
conda run -n rs python scripts/03_train_twotower.py
conda run -n rs python scripts/04_train_ranker.py
conda run -n rs python scripts/05_evaluate.py --max-eval-users 50
conda run -n rs python scripts/06_recommend_user.py --random_user --top_k 5
```

本地验证的目标是确认代码正确性，不追求最终指标。真正的训练质量需要在 GPU 服务器上放大数据量和训练轮数。

## 后续改进方向

- 加入 FAISS 做大规模向量检索
- 加入更完整的搜索意图特征
- 引入图像、多模态或文本预训练向量
- 加入更丰富的用户画像特征
- 增加排序模型校准
- 增加实验配置管理和指标对比报告
- 增加更细的重排策略与业务规则
