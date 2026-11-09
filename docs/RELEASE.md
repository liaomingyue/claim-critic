# 开源发布：数据、模型与复现

本页说明公开了哪些数据和模型、它们对应哪个结论，以及怎么从零复现。
- 结论本身见 [tech_share.md](tech_share.md)；
- 实验流水账见 [experiment_log.md](experiment_log.md)。

## 发布内容

数据和模型分成两部分，每部分按用途打成几个包。逐文件的 SHA-256 记在 `MANIFEST.json` 里。

### 数据：claim-critic-data

| 包 | 内容 | 对应什么 |
|---|---|---|
| `eval.tar.gz` | 三套验收回答（`outputs/rt/minicpm/grpo300{,b,c}/answers.jsonl`）+ Claude 逐句金标（`final.jsonl`）；各代 critic 在这三套上的逐句打分（`critic_<名字>.json`）；各版 RAG 在三套验收原文上的生成 + 9B 金标打分（`outputs/rt/core/loop2_*/eval/`）；奖励粒度 R0–R3 的生成和打分（`outputs/rt/minicpm/rwexp/eval/`） | 技术分享表 1–4、7–9 的全部数字（L1 复现） |
| `core.tar.gz` | 核心集：2B 写的原回答、Claude 逐句改对（`fix_*_raw.jsonl`）和错误类型 / 满意度标注（`label_raw.jsonl`），以及由它派生的数据（`core.jsonl`、`pairs.jsonl`、种子集、改写提示池等） | 第一到三步，所有模型的数据起点 |
| `synth.tar.gz` | 错误合成相关：<br>- 生成器 v2 输出的 Claude 标注（`gen_v2_label*`）；<br>- 改动描述；<br>- 难样本核实（`hard_check`、`blind_check`）；<br>- 两轮盲区补样本的提示池（`rag_pool*`）、采样、选样和 Claude 核实（`blind_fill`、`blind_fill_r2`）；<br>- 生成器 GRPO 的提示、试采和难样本 | 表 5–7、两轮闭环的 critic 数据 |
| `rm.tar.gz` | 各打分模型的训练 / 校准集和评测报告（`outputs/rt/rm/<名字>/`，`eval/report.json` 里有校准温度）。由 core 和 synth 派生，打包是为了方便直接训 | 直接重训任一 critic / sev / RM2 |
| `ragtruth.tar.gz` | RAGTruth 原件（MIT） | 原文池、人工标注底座 |

### 模型：claim-critic-models（只有 LoRA）

| 包 | 内容 |
|---|---|
| `critic.tar` | 各代 critic：rm1__core → r1 → r2 → r2_bfcl（第一轮）/ r2_bfnc（不核实对照）→ rd2（第二轮），以及表 3、4 里的各对照组；9B 金标 critic（rm1__rm1_gold_9b，只用于评测，基座是 Qwen3.5-9B） |
| `rag.tar` | RAG 各版本，见下方命名对照表 |
| `gen.tar` | 错误生成器：gen-v2-mc（SFT）、gen-v2h3-mc（难样本 SFT）、gen-g1（GRPO 50 步） |
| `rm.tar` | sev（v3c0、v3d）、RM2 自然度（v3b、v3d）、满意度（sat_pair__core）、拒答（refuse__core） |

RAG 版本的命名对照：

| 名字 | 说明 |
|---|---|
| rag-sft-core | 监督微调版 |
| RAG-S1 | 第 1 段强化 |
| RAG-S2 | 第 2 段强化 |
| RAG-S3-r2 / -r3 / -bf | 第 3 段的三个平行尝试，分别换用 r2 / r3 / bfcl 当奖励 |
| RAG-S3-r1 | 对照：奖励模型不换 |
| RAG-S4-rd2 | 第二轮闭环 |
| RAG-S1x | 对照：从 SFT 重新训 |

除 9B 金标 critic 外，基座都是 openbmb/MiniCPM5-2B（提交 f974000），我们不转发基座，请按其原许可证自行下载。

### 没有放进来的

- **RAGTruth++**（blue-guardrails/ragtruth-plus-plus）：它的数据卡没有写许可证，所以我们不转发它，也不转发由它派生的测试集（各 critic 目录下的 `test.jsonl`）。需要这个辅助口径的话，从原地址下载 `messages.csv` 和 `hallucination_spans.csv` 放到 `external_data/ragtruth_pp/`，然后运行 `python -m ragtruth.rtpp convert`。技术分享里的表 1–9 都不依赖它。
- **基座模型本体**：只发 LoRA。
- **Claude 的原始返回**（`labels_raw.jsonl`）：已经解析进 `labels.jsonl`。

### 许可

| 内容 | 许可 |
|---|---|
| 代码 | Apache-2.0（仓库根目录 `LICENSE`） |
| RAGTruth 原件 | MIT（随包附带） |
| LoRA 权重 | 遵循各自基座模型的许可证 |
| 我们新增的标注 | 由 Claude（claude-opus-5-5）生成，包括逐句改对、错误类型、核实、金标，以及 9B 生成的改动描述 |

## 复现

### L0：不需要 GPU，重算技术分享的表 1–9

```bash
git clone https://github.com/liaomingyue/claim-critic && cd claim-critic
pip install -r requirements.txt && pip install -e .
python -m scripts.demo_tables
```

数字来自仓库里自带的 `results/*.json`。这一层只验证"表是由这些结果算出来的"。

### L1：从原始逐句打分重算所有对比（不需要 GPU）

```bash
pip install huggingface_hub
python -m scripts.fetch_release --data_repo <用户>/claim-critic-data --only data --parts eval
# critic 对比（表 3、4、7，以及第二轮 critic 的静态评测）：
python -m ragtruth.critic_cap3 --critics rm1__core rm1__core_r2 rm1__core_r2_bfcl rm1__core_r2_bfnc rm1__core_rd2 --base rm1__core_r2 --by_type
# RAG 对比（表 2、8、9）：
python -m ragtruth.pol9b_cmp --arm s2=outputs/rt/core/loop2_cont/eval:s100 --arm s3bf=outputs/rt/core/loop2_cont4/eval:s100 \
    --arm s3r1=outputs/rt/core/loop2_cont5/eval:s100 --arm s4=outputs/rt/core/loop2_cont6/eval:s100 --base s2
```

这一层会从 Claude 金标和逐句打分重新算 AUC、召回和配对 bootstrap 区间，应该和 `results/` 里的一致。

### L1+：用发布的 LoRA 重新打分（需要单张 GPU）

```bash
pip install -r requirements-gpu.txt
python -m scripts.fetch_release --data_repo … --model_repo … --parts eval critic
# 起 vLLM，挂上要测的 critic LoRA，温度取自 outputs/rt/rm/<名字>/eval/report.json：
python -m ragtruth.critic_score --answers outputs/rt/minicpm/grpo300/answers.jsonl --model c --temp <T> --out …
```

RAG 的重新生成和 9B 打分，见 `scripts/remote/run_r3eval.sh`。

### L2：从核心集开始，跑完整闭环

```bash
python -m scripts.fetch_release --data_repo … --model_repo …     # 全部数据 + 模型
```

然后按 README 的"闭环"一节分步跑：
- RAG GRPO：`scripts/remote/run_loop2cont.sh`；
- 盲区补样本：`ragtruth/blind_fill.py`，第二轮的队列见 `scripts/remote/q_r2.tasks`；
- critic 重训：`scripts/remote/run_mcrm.sh`。

需要 Claude 接口的步骤只有两类：
- 核心集的改对和标注，这一步的结果已经在 `core.tar.gz` 里；
- 新一轮的盲区核实，已经做过的两轮在 `synth.tar.gz` 里。

所以不调用 Claude 也能完整重放两轮闭环：直接用发布的标签。

所有运行都固定了种子（`PYTHONHASHSEED=0` 加固定字符串种子）。GRPO 的生成用 vLLM，有少量非确定性：同一个检查点两次独立生成，含显然错的比例差约 1 个点。
