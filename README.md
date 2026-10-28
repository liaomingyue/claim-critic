# claim-critic

Claim 级医学综述幻觉检测 critic：给定一条中文 claim、它引用的英文参考文献（附同组对照文献和术语表），判定是否忠实，并给出逐字证据、错误类型和原因。用 GRPO 训练，reward = 标签 + 格式 + 定位。

错误分五类：citation_error 引用错误（含挂载错误和无中生有两种情况，靠证据区分）、term_error 术语翻译错误、wrong_attribution 张冠李戴、unsupported_inference 不合理推理、missing_condition 前提缺失。引用错误靠同组对照文献判断，术语错误靠自定义词典 `configs/glossary.tsv`。

方案设计见技术方案文档；**数据流程与各模块的职责见 [docs/pipeline.md](docs/pipeline.md)**。

## 流程

每个阶段读写 JSONL，可以单独重跑；调用模型的阶段支持断点续跑（已完成的 id 自动跳过）。

```bash
pip install -e ".[dev]"            # 数据流水线 + 单测
pip install -e ".[train,verl]"     # GPU 机器上

# 1. 数据：文献 → 综述 → claim → 正样本
python -m data.fetch_pmc --config configs/data.yaml --with_sections
vllm serve Qwen/Qwen3-32B --port 8000 &
python -m data.gen_review     --config configs/data.yaml
python -m data.split_claims   --reviews outputs/data/reviews.jsonl --out outputs/data/claims.jsonl --context_k 2
python -m data.label_positive --config configs/data.yaml        # 在审核台抽检 positives（错判率 ≤ 3%）

# 2. 冷启动：第 1 批合成负样本（未过滤）+ 正样本 → SFT → critic-0
python -m data.synth_negative --config configs/data.yaml --batch 1
python -m data.build_trainset --config configs/data.yaml --version 0
python -m train.sft --config configs/sft.yaml --mode coldstart

# 3. 三轮：critic-{r} 挖第 r+1 批 → 训练集 v{r+1} → 从 critic-0 起训 GRPO → critic-{r+1}
vllm serve outputs/ckpt/critic-0 --served-model-name critic-0 --port 8002 &
python -m data.mine_hard      --config configs/data.yaml --batch 1 --round 0   # 在审核台抽检 hard_batch1（not_an_error ≤ 10%）
python -m data.build_trainset --config configs/data.yaml --version 1
python -m train.grpo prepare  --config configs/grpo.yaml --version 1
python -m train.grpo launch   --config configs/grpo.yaml --version 1
#   把 grpo-v1 的 checkpoint 转成 HF 格式并命名为 critic-1，然后
python -m data.synth_negative --config configs/data.yaml --batch 2
python -m data.mine_hard      --config configs/data.yaml --batch 2 --round 1
#   …… 第 3 轮同上，得到 critic-3（发布版本）

# 4. 测试集人工标注
python -m annotation.ls_io export --claims outputs/data/test_claims.jsonl --out tasks.json
python -m annotation.ls_io import --ls_export project-a.json --out ann_a.jsonl   # B 同理
python -m annotation.kappa agree --a ann_a.jsonl --b ann_b.jsonl --disagree_out to_adjudicate.jsonl
python -m annotation.kappa merge --a ann_a.jsonl --b ann_b.jsonl --adjudicated adj.jsonl --out data_release/test.jsonl

# 5. 评测
python -m eval.run_eval --test data_release/test.jsonl --model critic-3 --out outputs/eval/critic-3
python -m eval.run_baselines --test data_release/test.jsonl --which minicheck --out outputs/eval/minicheck
python -m eval.ood --subsets RAGTruth --out outputs/ood/test.jsonl

# 推理
python -m infer.judge --input review.json --model critic-3 --out report.md [--check_uncited]
```

测试集的 claim 来自 `fetch_pmc` 划给 test 的文献组（`split_claims` 输出中 `split == "test"` 的行），抽约 1000 条送标注。

## 单卡冒烟测试（12 GB）

在正式上多卡之前，用 Qwen3-0.6B + LoRA 和演示数据把环境和训练流程跑一遍，只看能不能跑通，不看效果。Windows 可直接跑（GRPO 不用 vLLM）；vLLM、verl 只能在 Linux / WSL2 里装。

```bash
python -m venv .venv && .venv\Scriptsctivate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # 按驱动选 CUDA 版本
pip install -e ".[smoke,dev]"
# 缓存别放系统盘：PIP_CACHE_DIR、HF_HOME 指到数据盘；国内下载模型可设 HF_ENDPOINT=https://hf-mirror.com

python -m scripts.smoke_data                                   # 演示数据 → outputs/smoke/{coldstart.jsonl, trainset_v1/}
python -m scripts.env_check --model Qwen/Qwen3-0.6B-Base       # 包、GPU、配置字段是否被所装 TRL 认识、chat template 与长度
python -m train.sft --config configs/smoke/sft.yaml --mode coldstart          # → outputs/smoke/ckpt/critic-0（LoRA 合并后的完整权重）
python -m scripts.smoke_eval --model outputs/smoke/ckpt/critic-0             # 本地生成，查输出格式、reward、指标、显存峰值
python -m train.grpo_trl --config configs/smoke/grpo_trl.yaml --version 1    # GRPO 5 步
```

`env_check` 的 `[FAIL]` 必须修；最常见的是 TRL 改了字段名，按提示改对应 yaml。

已在 RTX 4070 SUPER 12 GB / Windows 11 / torch 2.11 + transformers 5.20 + TRL 1.14 上跑通：SFT 约 1 分钟，GRPO 5 步约 2 分钟。用 LoRA 训 Base 模型时，对话特殊 token（`<|im_end|>`、`<think>` 等）的词向量没训练过、LoRA 也碰不到，必须用 `lora_train_tokens` 放开，否则模型输出不了这些 token；`eos_tokens` 把 `<|im_end|>` 写进 generation_config，否则推理不会在回合结束处停（全参数训练同样需要）。

vLLM 服务和 verl 训练在 Windows 上用 Docker 跑，见 [docker/README.md](docker/README.md)。

## 人工审核台

迭代过程中模型产出的每一批样本都在这里看、改、拒，结果自动回流到训练集。只依赖标准库。

```bash
python -m review.server --config configs/data.yaml --reviewer 你的名字    # 打开 http://127.0.0.1:8765
python -m review.demo_data && python -m review.server --config configs/demo.yaml   # 没有数据时先用演示数据体验
```

- 数据集：positives、synth_batch{k}、hard_batch{k}、coldstart；视图：全部 / 抽检 / 未审 / 已修改 / 已拒绝，可按类型、轮次、关键词筛选。
- 每条样本显示：参考文献（被引用 / 对照标注、证据高亮）、与原 claim 的逐词差异和引用变化、术语表与词典预检命中的误译、生成模型的改写思路、critic 的 4 次采样输出、同一原 claim 在各批的派生样本、审核历史。
- 操作：通过 / 拒绝，打问题标记（类型标错、不是真错误、证据错位、强模型判错），直接修改 claim、引用（勾选参考文献）、label、type、证据、原因；在原文里划选文字可直接设为证据，无中生有的证据选 none。修改实时走与流水线相同的校验，校验不过不能保存为通过。
- 快捷键：`J`/`K` 上下条，`A` 通过，`R` 拒绝，`1`–`3` 切换标记，`Ctrl+Enter` 保存修改并通过；`?dataset=hard_batch1&id=...` 可直接定位到一条样本。
- 顶栏实时显示抽检子集上的问题比例与阈值（正样本 judge_wrong ≤ 3%，难样本 not_an_error ≤ 10%），超标标红。
- 审核记录追加写入 `outputs/review/<dataset>.jsonl`（同一样本以最后一条为准，历史保留）。`build_trainset` 读取时：拒绝的剔除，修改的用修改后的内容；原样本在审核后被重跑改动的，旧修改不套用，并在审核台提示重审。

## 可视化

```bash
pip install -e ".[app]"
python -m app.main --config configs/data.yaml --eval_dir outputs/eval --port 7860
```

- 「在线判定 Demo」：输入综述正文和参考文献，调用 vLLM 上的 critic 逐条判定；展开某条 claim 可看到证据在参考文献中的高亮位置。下方的「Reward 计算器」不需要模型，粘贴输出即可看到三项得分。
- 「数据与评测看板」：各阶段样本量、各批难样本的类型/轮次分布、检出率与重写次数分布、人工抽检比例（超过阈值标红）、样本浏览器、`outputs/eval/*/metrics.json` 的对比表（可选基线算 F1 绝对差与相对提升）。还没生成的文件显示「尚未生成」。

## 仓库结构

```
common/      样本格式、输出解析、prompt、对照文献选取、术语词典、LLM 调用（各阶段共用）
data/        fetch_pmc → gen_review → split_claims → label_positive → synth_negative → mine_hard → build_trainset
train/       reward.py（纯函数）、sft.py、grpo.py（verl）、grpo_trl.py（TRL + LoRA 备选）
eval/        metrics.py、run_eval.py、run_baselines.py、ood.py
annotation/  标注规范、Label Studio 配置与导入导出、κ 与裁决
infer/       judge.py：综述 → 逐条判定 → 报告
review/      人工审核台：store.py（存储、校验、回流）、server.py（标准库服务）、static/index.html
app/         Gradio 可视化：stats.py（纯函数）、main.py（界面）
configs/     各阶段 yaml；negative_examples.jsonl 为负样本示例池；glossary.tsv 为术语词典
docs/        pipeline.md：数据流程与模块说明
docker/      vLLM 服务与 verl 训练容器（Windows 用 Docker Desktop）
scripts/     环境检查与单卡冒烟测试（env_check、smoke_data、smoke_eval）
tests/       pytest
```

## 测试

```bash
python -m pytest -q
```

## 许可

代码 Apache-2.0；数据来自 PMC 开放获取子集中 CC-BY 许可的文章，再发布时保留出处；模型权重遵循 Qwen3 许可。
