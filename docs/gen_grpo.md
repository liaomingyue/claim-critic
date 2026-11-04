# 错误生成器 GRPO：奖励设计与训练方案

2026-10-31。代码：`train/gen_reward.py`（奖励）、`train/gen_grpo.py`（训练 / 快速评测）、`ragtruth/gen_grpo_data.py`（prompt 池）。

## 1. 目标

给定一条（人工标注全对的）RAG 回答，生成器改写其中一句或插入一句，造出一个**合格、难、像自然犯错、多样**的幻觉样本
（类型：conflict 矛盾 / baseless 无依据），用于训练 critic。起点是 G_dpo2（G_seed → DPO 两轮，拼接成一个 r32 LoRA）。

## 2. prompt 池（不需要标准答案）

- 6,000 条 prompt = 3,000 条回答 × 2 套示例；回答来自 RAGTruth 训练集人工标注全对、≥ 2 句的回答，1,605 篇原文。
- 按（任务, 生成回答的模型）分层轮流抽，每篇原文最多 3 条；原文不和评测用的 198 条种子回答重叠。
- 每个 prompt 重新抽 3 个真实幻觉示例（同任务、不同原文）；打乱后只过一遍，同一 prompt 不重复。
- 正式训练 300 步 × 16 prompt = 4,800 条，不会用到重复的 prompt。

## 3. 奖励（每条采样）

| 情况 | 奖励 |
|---|---|
| 硬规则没过 | 0 |
| 过规则，RM3 P(合格) < 0.5 | 0.1 × P(合格)（≤ 0.05，比任何合格样本低，但保留一点梯度） |
| 过门槛但原句被 RM1 判有幻觉（P > 0.5） | 0（2026-10-31，用户；已审 200 条上只抓到 Claude 判原句有问题的 5/19，误伤 4/181） |
| 合格 | (w_hard·难度 + w_real·真实性 + w_div·多样性) / (w_hard + w_real + w_div)，默认 1 : 1 : 1 |

### 3.1 硬规则（防 reward hack / 走捷径）

| 规则 | 防的是什么 |
|---|---|
| 解析不出 JSON / 句号越界 / 类型不是 conflict、baseless / 改写句为空 | 无效输出 |
| 改写句和原句相同（规范化后） | 什么都不改拿分 |
| 整句照抄回答里的另一句 | 复制粘贴拿分 |
| 改写句 > 原句 3 倍且多出 300 字符以上 | 一句里塞好几个错，骗过 RM |
| 改写句出现 hallucination / faithful / the source / source text / not stated (mentioned) in | 明说"这里有错"，骗 RM3 |

格式规则本身不是目标（用户：不在意规则是否遵守，只注重多样性），这里只拦"根本没造出有效错误"和明确的作弊方式。

### 3.2 三项分（都在 [0, 1]）

- **难度**：`4p(1−p)`，p = RM1 P(有幻觉) = P(B 矛盾) + P(C 无依据)。p = 0.5（critic 拿不准）最高；
  p→1（一眼看出）或 p→0（RM1 认为不算错）都低。
- **真实性**：RM2（rm2v2）P(A 自然)，像目标模型自然犯的错。
- **按类型标准化**：难度、真实性都在同一错误类型内用滑动均值 / 方差（指数平均 decay 0.99，前 20 条 0.8）标准化后过
  sigmoid。类型之间不互相挤压（沿用 DPO 第二轮"按类型分位"）。
- **多样性**：`1/√(1 + N(类型, 改法))`，N 是**最近 512 条**合格样本里这一类的次数（用户选方案 A：按实时分布算。累计计数到第 50 步就只剩约 0.06，等于没有这一项）。改法封闭到 `synth.TACTICS` 的标准清单（conflict 7 种、baseless 6 种），对不上的归 `other`，`other` 多样性记 0（`gen_reward.closed_tactic`，见 3.4 第 3 条）；
  某类型在最近 512 条合格样本里占比 < 0.3（type_floor）时，这类样本多样性记 1。

三个 RM 都是同一 9B 底座上的 LoRA，读单字符选项概率，用各自的全局温度校准
（RM1 T=1.3，rm2v2 T=1.15，RM3 T=1.65）。

### 3.3 优势值

同一 prompt 的 n=8 条输出：每条减去**同类型**输出的平均奖励（同类型只有它自己时减全组均值），除以全组标准差；
全组奖励相同则优势全 0（不贡献梯度）。对应 DPO 第二轮"chosen 和 rejected 必须同类型"。

### 3.4 已知的性质（需要知道的）

1. **type_floor 兜底基本不起作用**：同类型样本一起被记满多样性，减同类型均值后抵消；只在一组里某类型只出现 1 次时有效。
   反过来，"选哪种类型"本身也不受奖励推动，不会因奖励偏向某一类；防偏主要靠 KL，训练中监控 conflict_share。
2. **多样性按窗口算后不再单调下降**。改法名字仍是模型自填，模型可能把冷门的标准改法名硬贴到错误上骗加分，所以评测要同时看不依赖自填内容的 `distinct_place_type@K`。看效果用原始分项（log.jsonl 的
   p_halluc / p_natural / hard_share / qualified_rate）和每个检查点的快速评测。
3. **已修的 reward hack（2026-10-31）**：第一版改法是开放字符串，对不上标准名时保留原文。模型第 50–100 步学会了编新名字（"a preheating step"…），
   每个新名字 N=0、多样性满分：快评改法 18 → 71 种，div 反升 0.12 → 0.22，而不看改法的不同句子数、两两相似度基本不变。这轮（full_share）在第 142 步停掉。
   教训：模型自填的字段不能直接进奖励。评测同时看 `distinct_place_type@K`（只按 op、句号、类型去重）和 `other_tactic_share` / `raw_tactics_used`。

## 4. 训练

- 两张 RTX 5090：GPU0 训练（HF + PEFT，只训生成器 LoRA）；GPU1 一个 vLLM 服务同时挂 RM1 / RM2 / RM3 和生成器 LoRA，
  训练中热加载新版本（`/v1/load_lora_adapter`），保留最近 3 个版本。
- 损失：PPO 截断（clip 0.2）的重要性比值（相对 vLLM 采样时的 logprob）× 优势值 − β·k3 KL（β = 0.05，参考 = 起点
  G_dpo2，参考 logprob 在 vLLM 上用 prompt_logprobs 算）。
- 异步：生产者（采样 + RM 打分 + 参考 logprob）一直往前跑，训练用的批最多落后 2 个版本，超过的丢掉重采。
- 正式参数：300 步，每步 16 prompt × 8 采样，max_tokens 768，temperature 1.0，lr 5e-6，micro_batch 2，每 50 步存检查点。

性能复测（8 prompt × 8，3 步）：每步约 60–70 s；两卡同时忙 56%（优化前异步 35%、同步 0%）；micro_batch 4 显存峰值
31.3 GB 太紧，正式用 2（26.9 GB）。

## 5. 检查点快速评测

每个检查点存完，在推理卡上后台评一次（不挡训练），开训前先评起点（step 0）：

- 评测集：生成器评测的 198 条种子回答（原文和训练 prompt 不重叠），示例固定（同 `ragtruth.gen_eval sample`），每条采 4 次。
- 只打 RM 原始概率（`Rewarder.raw`，不动训练的滑动统计和多样性计数）；检查点另起名字加载，评完卸载。
- 指标（口径同 `ragtruth.gen_eval report`）：硬规则通过率、合格率、不同合格错误数 @1/2/4（按 op, 句号, 类型, 改法去重）、
  每条回答不同句子数 / 改法数、两两相似度、类型分布与 conflict 占比、insert 占比、RM1 P(有幻觉) 均值与 hard_share
  (<0.5)、RM2 P(自然) 均值与 natural_share (>0.5)、截断数。
- 输出：`outputs/rt/realgen/grpo/full/quick_eval.jsonl`。训完后对选中的检查点再做完整评测（198 × 16）。

## 6. 输出

`outputs/rt/realgen/grpo/full/`：`log.jsonl`（每步奖励各项、合格率、类型占比、硬规则拦截原因、落后版本数、截断比例、KL、
耗时）、`gpu.csv`（两卡利用率，`scripts/grpo_gpu_report.py` 分析）、`ckpt/step{50,100,…,300}/`、`quick_eval.jsonl`。

## 7. 监控和回退（2026-10-31，用户：不能每次 reward hack 都重训）

**自动报警（`train.gen_grpo.TripWire`，默认开）**：第 45 步起，最近 20 步均值和第 6–25 步的基线比，越线就存状态、写 `TRIP.json`、打印 `GRPO_TRIPPED` 并停。

| 信号 | 线 | 防什么 |
|---|---|---|
| `other_tactic_share` | ≥ 8% | 编改法名字 |
| `raw_tactics` | ≥ 基线 2 倍且多 10 | 同上 |
| `conflict_share` | 不在 15–60% | 类型塌缩 |
| `qualified_rate` | 比基线低 12 个点 | 走偏 |
| `top_tactic_share` | ≥ 60% | 集中到一种改法 |
| `kl` | ≥ 0.1 | 离起点太远 |
| `reason_chars` / `evidence_chars` / `sentence_chars` | ≥ 基线 1.5 倍 | 堆字骗 RM（RM3 能看到模型自填的理由和证据） |

快评里按改法去重的多样性比第一次涨 ≥ 10%、不看改法的涨 ≤ 2% 时打印 `QUICK_EVAL_WARN label_gaming`（贴标签），只报不停。

**报警抓不到的**：句子本身骗过 RM（奖励和这些指标都由 RM 决定），要靠不进奖励的独立裁判抽检（如 Claude 盲审），`p_correct` 上涨而合格样本质量没涨就是这种。

**保存的内容**：`samples/stepN.jsonl.gz` 每步全部原始输出和打分；`state/stepN/` 每 25 步完整状态（LoRA、优化器、奖励统计、prompt 位置、随机数，约 1 GB，默认全留）；`ckpt/stepN/` 每 50 步 LoRA；`log.jsonl`、`gpu.csv`、`quick_eval.jsonl`、`quick_samples_*.jsonl`。

**回退**：改完奖励后 `--resume <out>/state/stepN`；奖励口径变了加 `--reset_reward 1`。

## 8. 分支 full_v3：奖励换位置（2026-10-31，用户）

主线第 100 步的问题：看改法的多样性涨 9%，不看改法的只涨 1%，改到的句子数反而降 5%，模型在差不多的位置上换标签。分支从主线第 200 步开始，`--div_mode place`：同一 prompt 这组合格样本里，同一位置（操作, 句号）出现 n 次，每条多样性 1/√n。类型和改法都是模型自填，不进这一项；改法归 `other` 仍记 0。主要看 `distinct_sentences_per_resp`、`place_per_prompt`，同时看难度和真实性有没有掉。
