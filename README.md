# claim-critic（RAGTruth 主线，loop-v1）

RAG 回答的逐句幻觉检测 critic，加两条循环：错误改写模型造边界错误来扩充 critic，critic 再当奖励训练 RAG 回答模型。

整个项目只有一个起始数据集：**2B 自己写的 RAG 回答 + Claude 逐句改对和标注**（下称核心集）。其余数据集都从它派生，训练用的模型全部是 MiniCPM5-2B（下称 2B）。

> 本文是待确认的设计稿（2026-11-06），逐节确认的网页版：https://claude.ai/artifact/NneTnoimA1aoWUZZxgvqNE 。
> 标「已有」的是现有代码和数据；标「新增」的要按本文新写或新标注。

- 提示词原文：[docs/prompts.md](docs/prompts.md)（`python -m scripts.dump_prompts` 从代码导出）
- 结果与结论：[docs/results.md](docs/results.md)；完整实验记录：[docs/experiment_log.md](docs/experiment_log.md)
- 金标标注规范：[docs/gold_standard.md](docs/gold_standard.md)；生成器 GRPO 设计：[docs/gen_grpo.md](docs/gen_grpo.md)

## 总览

| 角色 | 模型 | 用在哪 |
| --- | --- | --- |
| 训练的全部模型 | MiniCPM5-2B + LoRA | 回答、错误改写、critic、各打分模型 |
| 评测和监控 | Qwen3.5-9B 金标 critic | 只打分，不进奖励，不筛训练数据 |
| 标注和验收 | Claude（claude-opus-5-5） | 核心集标注、sev 单独标注、最终验收 |

```
第一步 2B 写回答 ──► 第二步 Claude 改对 + 标注 ──► 核心集 ──► 第三步 派生 D1–D9 ──► 第五步 训练各模型
                                                              │
                         第四步 sev 单独标注（改写模型 v0 的输出）┘
循环一（错误合成）：改写模型 GRPO ─► 采集边界错句 ─► critic 重训 ─┐
循环二（RAG 生成）：RAG SFT ─► GRPO（critic + 满意度 + 拒答 + 硬规则）◄┘
```

每个结果都写明阶段（RAG 生成 / 错误合成 / critic 评测）、数据来源和模型。全部运行 `PYTHONHASHSEED=0` + 固定种子，按原文切分用固定字符串种子，重跑得到同样的数据。

## 第一步：选 base 模型、调提示词、生成回答（RAG 生成，已有）

结论：base 模型是 MiniCPM5-2B，关闭思考，QA 加一句系统提示；在 831 个训练原文上得到 807 条回答。

**1.1 base 模型**

| 项 | 取值 |
| --- | --- |
| 模型 | openbmb/MiniCPM5-2B，提交 f974000，权重 SHA-256 核对过 |
| 初始策略 | `mc-init`：零初始化 LoRA，输出等于基座 |
| 思考 | 关闭（`enable_thinking=False`）。开思考每条约 813 token，关闭约 81 token |

**1.2 调提示词**

- 问题：关闭思考时 QA 几乎全拒答，100 个原文里的 60 道 QA 拒答 53–54 条（9B 只有 1 条）。
- 做法：QA 加系统提示 `ragtruth.ans_data.SYSTEM_QA`：有相关信息就给出能支持的最好回答，哪怕不完整；只有完全没有相关内容才拒答。拒答降到 14 / 60。
- Summary 不加系统提示，user 消息就是 RAGTruth 原 prompt（任务说明 + 原文）。

**1.3 原文池**：RAGTruth 官方 train 划分的 QA 和 Summary（没有 Data2txt：验收集只有这两类），排除评测、校准、金标种子、dev 的原文和超过 8000 字符的原文。`python -m ragtruth.ans_data split` → 831 个（QA 433 / Summary 398），`outputs/rt/ansgrpo/train_all.jsonl`。

**1.4 生成回答**：`python -m ragtruth.ans_data ref2 --model policy_v0 --sys_qa 1`。温度 0.7，每个原文采 4 条，最长 1024 token，取正常结束、不拒答里长度居中的一条。结果 807 条、3740 句 → `outputs/rt/minicpm/grpo/train_ref.jsonl` 的 `ref_text`。缺的 24 个全是 QA（4 条都拒答或没结束），保留为 GRPO prompt，不进核心集。

## 第二步：Claude 改对并标注 → 核心集

**2.1 逐句改对（已有）**

- claude-opus-5-5，effort high，提示词 `ragtruth.policy_sft.FIX_SYSTEM`（严格口径：拿不准就算错）。
- 只改错句，最小改动，保持它在回答里的作用、长度和风格；改不了就删；该答却拒答的整条重写。
- 每次 10 条，81 次批量 + 7 次单条补漏：`python -m ragtruth.policy_sft s1_fix` → `outputs/rt/minicpm/s1/fix_raw.jsonl`、`fix_single_raw.jsonl`。

| 结果 | 条数 |
| --- | --- |
| 有改动的回答 | 597（共改 1680 句） |
| 全对的回答 | 195 |
| 整段重写 | 15 |
| 每条改几句 | 0 句 195 / 1 句 189 / 2 句 116 / 3 句 89 / ≥ 4 句 203 |

参考（critic 评测，只看不用）：9B 金标 critic 打分，显然错从每条 0.913 句降到 0.099 句。

**2.2 错误类型和满意度标注（新增，约 81 次 Claude 调用，跑前确认）**

输入原文、原回答编号句子和 2.1 的改对句。Claude 不重新判哪句错，只标属性。提示词 `ragtruth.core_label.LABEL_SYSTEM`（草稿）。

每个改动（错句）：

| 字段 | 取值 | 用在哪 |
| --- | --- | --- |
| severity | clear 显然 / minor 细微 | sev 数据、种子分层 |
| type | conflict 与原文矛盾 / baseless 无中生有 | critic 标签 B / C、种子 |
| mechanism | wrong_attribution 张冠李戴、near_synonym_shift 偷换概念、certainty_or_scope_shift 确定性或范围变化、number_or_date 数字日期、merged_across_passages 跨段拼接、added_specific_fact 添加具体事实、vague_evaluation_or_addon 添加评价、unsupported_inference 无依据推断、contradicts_source 其他矛盾、other | 分类型统计、改法 |
| evidence / reason | 错句涉及的原文原话；一句理由（≤ 30 词） | 种子的 evidence / reason |
| fixed_evidence / fixed_reason | 支持改对句的原文原话；理由 | 种子的 original_evidence / original_reason |

每条回答（原回答、改对后各一次）的满意度：answers_task（yes / partial / no）、completeness（complete / missing_some / missing_much）、style（ok / problem）、sentence_form（ok / problem）、overall（good / not_good）、note。满意度不管幻觉，口径同 `SAT_PAIR_SYSTEM`。

有了 2.2，种子的类型、证据、理由直接来自 Claude，不再借旧生成器 gen-seed5-mc 补全。

**2.3 对齐和切分（程序，新增 `ragtruth.core_build`）**

- 切句 `ragtruth.convert.split_sentences`，和 Claude 看到的编号一致。每处改动得到一对：原错句、改对句；上下文是全部改动应用后的句子列表。
- 按原文留出 10%（种子字符串 `core|holdout`，81 个原文），只用于评测。

| 去掉 | 数量 | 原因 |
| --- | --- | --- |
| 整段重写 | 15 条 | 句子对不上 |
| 删句 | 22 处 | 生成器只有改写，没有插入 |
| 上下文只剩一句 | 27 处 | 生成器要求至少两句 |
| 改对句拆成多句 | 4 处 | 一对多 |

| 切分 | 原文 | 可对齐回答 | 错误对 | 没改的句子 |
| --- | --- | --- | --- | --- |
| 训练 | 726 | 713（QA 355 / Summary 358） | 1472（Summary 1255 / QA 217） | 1835 |
| 留出 | 81 | 79（QA 39 / Summary 40） | 155（Summary 138 / QA 17） | 206 |

产出：`outputs/rt/core/core.jsonl`（按回答）、`outputs/rt/core/pairs.jsonl`（1627 对）。

## 第三步：由核心集派生的数据集（新增 `ragtruth.core_derive`）

「更多样的回答」= RAGTruth 里 6 个模型（GPT-4、GPT-3.5、Mistral-7B、Llama-2 7B / 13B / 70B）在**同一批 831 个原文**上的回答，共 4986 条，其中没有人工幻觉标注的 3432 条。81 个留出原文在每个数据集里都只进评测。带「约」的条数要跑完才知道。

| # | 数据集 | 怎么来 | 训练 | 评测 | 谁用 |
| --- | --- | --- | --- | --- | --- |
| D1 | RAG SFT | prompt = 原文（QA 加系统提示），目标 = 改对后的回答（全对的用原回答，整段重写的用重写） | 726 | 81 | RAG SFT |
| D2 | RAG GRPO prompt 池 | 831 个原文去掉 81 个留出；参考回答用 RAG SFT 模型重采 | 750 | 验收集 | 循环二 |
| D3 | 错误改写种子 SFT | 输入 = 改对后的回答，输出 = 把改对句改回原错句，类型 / 证据 / 理由取自 2.2 | 1472 | 155 | 改写模型 v0 |
| D4 | 错误改写 prompt 池 | 改对后的回答（至少两句）× 两种类型；加更多样回答里没有人工幻觉标注的 | 657 × 2 = 1314，另约 3000 条多样回答 | 70 | 循环一 |
| D5 | RM2 自然度 | B = 改写模型 v0 在更多样回答上的改写句（过硬规则）；A = 同位置原句 + 核心集真错句 1472；不用人工标注 | 约 6000 | 按原文留出 | 循环一奖励 |
| D6 | sev | 核心集派生 A–D + 单独标注（第四步） | 约 3700 | 单独标注的 20% | 循环一门槛 |
| D7 | 满意度（成对） | 2.2 满意度配对 + 程序反例（截断、照抄原文、只留第一句、该答却拒答），A / B 两种顺序 | 约 2500 | 留出派生 | 循环二护栏 |
| D8 | 拒答 | 只有 QA：原题 = 能答；删掉支持段落 = 答不了（按重合度）；问题配另一篇无关原文 = 答不了 | 约 900 | 留出 QA 39 篇派生 | 循环二 |
| D9 | critic（RM1） | 真错句 1472（B / C）+ 同位置改对句 1472（A）+ 没改的句子 1835（A） | 4779 | 校准：留出 516 句 | 循环二奖励 |
| D9+ | critic + 采集 | D9 + 循环一每轮采集的边界错句和同位置原句 | 约 4779 + 1000 / 轮 | 同 D9 | 循环二奖励 |

D8 删段落：RAGTruth 的 QA 原文都是 3 段。删掉改对回答用到的段落，剩下的重新编号，"three passages" 改成 "two passages"。改对回答和剩余段落的词重合低于门槛 → 答不了，否则 → 能答。门槛用旧的 40 条 Claude 复判过的删段样本定（旧做法 `ragtruth.refuse_data` 每条都要 Claude 复判）。

## 第四步：需要单独标注的数据（RM3 / sev）

sev 判「改写是不是真错、类型和解释对不对」。核心集只有真错，教不了「不是错」和「原句已错」，所以要对改写模型 v0 的输出单独标一批。

| 选项 | 含义 | 生成器奖励里怎么用 |
| --- | --- | --- |
| A | 显然错，解释对 | P(A) + P(B) ≥ 0.5 才合格 |
| B | 显然错，解释不对 | 同上 |
| C | 细微错，解释对 | 不合格，拿保底分 |
| D | 细微错，解释不对 | 不合格 |
| E | 不是错 | 不合格 |
| F | 原句本来就错 | P(F) > 0.5 记 0 分 |

- 来源一，核心集派生（程序）：改写句 = 原错句，原句 = 改对句，类型 / 证据 / 理由取自 2.2；显然 → A，细微 → C。同一句把类型换成另一种或证据换成同原文另一对的 → B / D。共 2944 行，只进训练。
- 来源二，单独标注（新增，约 40 次 Claude 调用，跑前确认）：改写模型 v0 在 D4 训练池上的输出，过硬规则的按类型和 critic 分数分层抽约 800 条，每个原文至多 2 条；提示词 `ragtruth.sev_label.SYSTEM`，每次 20 条；原句不对 → F，否则按 severity 和 explanation_ok 得 A–E；按原文 70 / 10 / 20 切训练 / 校准 / 测试，测试只用这部分。
- 旧的 sev 标签（787 + 504）和 RM3 标签（4358）来自旧生成器和另一批回答，不进新循环，只用来对比。旧 sev__mc 测试 AUC 0.805，是当前最弱的一环。

## 第五步：训练各模型

打分模型都是「单字母选项」：只读答案位置各选项的 logprob，在校准集上拟合一个全局温度，输出校准后的概率。

| 模型 | 数据 | 配置 | 输出 | 新名字 |
| --- | --- | --- | --- | --- |
| 错误改写模型 v0 | D3 | `configs/rt/sft_gen_seed5_mc.yaml`：LoRA r16 / α32，all-linear，1 轮，lr 5e-5，批 2 × 累积 2，最长 4096 | JSON：plan、idx、type、tactic、sentence、evidence、reason、original_evidence、original_reason | gen-core-mc |
| RAG SFT | D1 | `configs/rt/sft_policy_mc.yaml`：LoRA r32 / α64，7 个线性层，2 轮，lr 5e-5 | 回答文本 | rag-sft-core |
| critic（RM1） | D9 | `configs/rt/rm_mc.yaml`：LoRA r16，2 轮，lr 1e-4，批 4 × 累积 4 | A 忠实 / B 矛盾 / C 无依据；P(错) = P(B) + P(C) | rm1__core |
| RM2 | D5 | 同上 | A 自然 / B 合成 | rm2__core |
| sev | D6 | 同上 | A–F | sev__core |
| 满意度 | D7 | 同上 | A / B 哪条更好 | sat_pair__core |
| 拒答 | D8 | 同上 | A 能答 / B 答不了 | refuse__core |

- 打分模型：`RM=<名字> GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh`（训练、合并、拟合温度，写 `outputs/rt/rm/<名字>/eval/report.json`）。生成类：`python -m train.sft --config <配置> --mode coldstart`。
- 顺序：先训改写模型 v0、RAG SFT、critic、满意度、拒答（只依赖核心集）；再用改写模型 v0 采样，造 RM2 负样本和 sev 待标样本，然后训 RM2、sev。

## 循环一：错误合成（改写模型 GRPO → 采集 → critic 重训）

代码 `train/gen_grpo.py` + `train/gen_reward.py`（`TargetRewarder`，`shape=sev`），脚本 `scripts/remote/run_gensev.sh`。

1. 起点：第 1 轮从 gen-core-mc，之后从上一轮的检查点接着训。
2. GRPO：D4 训练池，每步 16 个 prompt × 16 条，lr 5e-6，50 步；快评不好就再续 50 步。
3. 快评：留出 70 个 prompt × 4 次，看解析成功率、合格率、critic 分数分布、自然度。
4. 采集：最后的检查点在 D4 训练池上每个 prompt 采 8 条（`gen_target_data harvest --sev ... --gate 0.5`）。
5. 入库：`critic_harvest --lo 0 --hi 1.01 --gate 0.5 --min_nat 0.3`，每轮最多约 550 对；每对是错句（B / C）+ 同位置原句（A）。
6. 抽检（可选）：Claude 抽 100 对，在后台跑，不停训练。
7. 重训 critic：D9 + 历轮采集 → `rm1__core_r<轮次>`，给循环二用。

奖励（P_ok = sev 的 P(A) + P(B)，P_F = sev 的 P(F)，p = critic 对改写句的 P(错)，q = RM2 的 P(自然)，d = 多样性）：

```
r = 0                                                 没过硬规则，或 P_F > 0.5
r = b · P_ok / 0.5                                    P_ok < 0.5（保底）
r = h(p) · (L(q) + w_d · d) / (1 + w_d) + bonus · 1[l ≤ p ≤ u]     其余
```

- 硬规则：能解析出 JSON；句号在范围内；类型是 conflict / baseless，指定了类型就必须一致；改写句不等于原句，也不等于回答里另一句；不超过原句 3 倍且多出不超过 300 字符；不出现 hallucination、faithful、"the source" 这类点破题目的词。
- L(q)：对数几率线性映射到 [0, 1]（±8 截断，`logodds01`）。h(p)：普通模式 1 − p；区间模式 1 − |p − 0.5| / 0.5。d：同一 prompt 16 条里同一句号被改 n 次，每条 1 / √n。
- 优势：同一 prompt 内减同类型的均值，再除以组内标准差。

| 参数 | 前 50 步 | 续训 50 步（区间模式） |
| --- | --- | --- |
| 保底 b | 0.1 | 0.02 |
| w_d | 0.1 | 0.1 |
| bonus，[l, u] | 0 | 0.5，[0.3, 0.7] |

进奖励：sev__core（门槛）、rm1__core（critic）、rm2__core（自然度）、硬规则。只监控：无（9B 不进奖励、不筛样本）。本轮合不合格看 critic 重训后「显然错召回」有没有涨。

## 循环二：RAG 生成（回答 GRPO，R3 奖励）

代码 `train/ans_grpo.py` + `train/ans_reward.py`（`--reward_shape count`），脚本 `scripts/remote/run_r3full.sh`。

1. 起点 rag-sft-core（对照组从 mc-init）。
2. 参考回答：起点模型在 D2 上重采（`ans_data ref2`），定长度范围、满意度参考和 K。
3. GRPO：D2，每步 16 个 prompt × 16 条，lr 5e-6，200 步，第 50 / 100 / 200 步存检查点；单卡，策略 vLLM 占 0.35 显存，RM 挂在同一服务上；TripWire 打开。
4. 评测：各检查点在验收集上生成，9B 金标 critic 打分，算被钻空子的程度。

奖励（p_i = critic 对第 i 句的 P(错)；b、σ = 同一 prompt 全部回答全部句子的 P 均值和标准差，σ 下限 0.1；e_i = max(0, p_i − b)；n_ref = 参考回答句数）：

```
H = 1 − 0.5 · (1 − Π_i (1 − e_i)) − 0.5 · min(1, Σ_i e_i / K)，  K = max(2, 0.5 · n_ref)
r = 0                                   没过硬规则
r = 0.4 · P(可拒答)                      拒答（只 QA）
r = max(0, H − 0.8 · max(0, 0.5 − P_sat))   其余
```

- 去底噪：只算超出同组平均的部分，否则 critic 对正确句的底噪（平均 0.10）按句数累加，逼模型少写。
- 计数：多一句错多扣一份；K 用参考回答的句数，多写安全句稀释不了。
- 句子层优势：整段优势之外，第 i 句的 token 再加 β × (−e_i / σ)，β = 0.6；后一句的负值按 γ = 0.3 传给前一句。
- 满意度只罚不奖：P_sat = sat_pair 判「新回答比参考回答好」，低于 0.5 才扣，最多扣 0.4。
- 硬规则（没过记 0）：空回答或截断；长度在参考回答字符数的 0.5–1.5 倍之外；照抄原文（连续 6 词在原文里的占比 QA ≥ 0.9、Summary ≥ 0.5）；结尾没有句末标点；原文没有中文而回答夹中文。

进奖励：rm1__core_r<轮次>、sat_pair__core、refuse__core、硬规则。只监控：9B 金标 critic（rm1__rm1_gold_9b）。

| 组 | 起点 | critic |
| --- | --- | --- |
| 主组 | rag-sft-core | rm1__core_r1 |
| 只核心集 | rag-sft-core | rm1__core（D9） |
| 只 SFT | rag-sft-core | 不做 GRPO |
| 旧基线 | mc-init | rm1__mcgold_h3ctl（已有结果 r3full） |

## 评测

三个阶段各有固定的评测集，全部和训练原文不重叠。评测里不调 Claude；最终验收要用 Claude 时先确认。

| 数据 | 内容 | 规模 | 标签 | 文件 |
| --- | --- | --- | --- | --- |
| 验收集 a | 2B GRPO 第 300 步在验收原文上的回答，温度 1.0 | 100 条 / 68 原文 / 414 句（显然错 30） | Claude 金标 v1 逐句 | `outputs/rt/minicpm/grpo300/{answers,final}.jsonl` |
| 验收集 b | 同上，温度 0.7 | 100 条 / 100 原文 / 386 句（显然错 45） | 同上 | `outputs/rt/minicpm/grpo300b/` |
| 验收集 c | 剩余没用过的 RAGTruth 测试原文 × 3 种子，温度 0.7 | 396 条 / 132 原文 / 1722 句（显然错 164） | 同上 | `outputs/rt/minicpm/grpo300c/` |
| RAGTruth++ | RAGTruth 测试集重标版 | 校准 427 句、测试 2306 句 | 人工 ++ 标签 | `outputs/rt/rm/rm1/{calib,test}.jsonl` |
| 核心集留出 | 81 个原文 | 错误对 155、没改句 206、改写 prompt 70 | Claude 改对 + 2.2 | `outputs/rt/core/`（新增） |
| sev 测试 | 单独标注的 20% | 约 160 | Claude 6 选 1 | 新增 |

验收的 596 条回答只定义「原文 + 种子 + 温度」，每个策略在同样的原文和种子上重新生成（`ragtruth.mc_accept_gen`），按回答 id 配对。金标提示词 `ragtruth.gold.SYSTEM`。

- **RAG 生成**：9B 金标 critic 逐句打分，p ≥ 0.9 记显然错。每条回答的指标：含显然错比例、显然错句数、含任意错（p ≥ 0.5）、拒答率、字符数、句数。按原文聚类 bootstrap（2000 次，90% 区间）配对比：`ragtruth.pol9b_cmp`。被钻空子：只看 9B 判显然错的句子，比较奖励 critic 在起点回答和第 200 步回答上的召回。最终验收（可选）：Claude 金标标同一批 100 个原文-种子（`ragtruth.accept_compare`）。
- **critic 评测**：主口径 `ragtruth.critic_cap3`：三套验收回答的 Claude 金标，显然错 vs 没错，去掉细微错和拒答句，共 2061 句、显然错 179；AUC（90% 区间）、误报 10% / 5% 时的召回、和基线的配对差，召回优先。辅助：RAGTruth++ 测试 2306 句、核心集留出 516 句。
- **错误合成**：

| 对象 | 数据 | 指标 |
| --- | --- | --- |
| 改写模型 | 留出 70 个 prompt × 4 次 | 解析成功率、过 sev 门槛比例、critic 分数分段（< 0.3 / 0.3–0.7 / > 0.7）、RM2 判自然比例 |
| sev | sev 测试 | 显然 vs 其余 AUC；门槛 0.5 时入库样本里真错的比例 |
| RM2 | 按原文留出 | 自然 vs 合成 AUC；核心集真错句被判自然的比例 |
| 满意度、拒答 | 各自留出 | 准确率、AUC、校准误差 |

## 提示词

全部原文在 [docs/prompts.md](docs/prompts.md)，由 `python -m scripts.dump_prompts` 从代码导出。

| 阶段 | 提示词 | 常量 |
| --- | --- | --- |
| RAG 生成 | QA 系统提示 | `ans_data.SYSTEM_QA` |
| 核心集 | 逐句改对（已跑） | `policy_sft.FIX_SYSTEM` |
| 核心集 | 错误类型和满意度（新增，没跑） | `core_label.LABEL_SYSTEM` |
| 单独标注 | sev 标注 | `sev_label.SYSTEM` |
| 单独标注 | 拒答判定 / 改题（新方案只用来定门槛） | `refuse_data.SYSTEM` / `REWRITE_SYSTEM` |
| 错误合成 | 改写 prompt 模板 + 改法清单 | `synth.FREE_PROMPT` + `FOCUS_TACTICS` |
| 打分模型 | critic / 自然度 / 严重程度 / 满意度 / 拒答 | `rm_data.RM1_SYSTEM` / `rm_data.RM2G_SYSTEM` / `sev_rm.SEV_SYSTEM` / `sat_build.SAT_PAIR_SYSTEM` / `refuse_data.RM_SYSTEM` |
| 评测 | 金标逐句标注 | `gold.SYSTEM` |

种子 SFT 一行：prompt = `FREE_PROMPT` 填入原文和**改对后**的编号句子（按 id 的 crc32 一半指定类型；示例句从训练池其他原文的真错句抽）；completion =

```json
{"plan": "Introduce one conflict error into S3: attribute a statement to the wrong person.", "idx": 3, "type": "conflict",
 "tactic": "<按 2.2 机制映射到 FOCUS_TACTICS，对不上写 other>", "sentence": "<原错句>",
 "evidence": "<2.2 evidence>", "reason": "<2.2 reason>", "original_evidence": "<2.2 fixed_evidence>", "original_reason": "<2.2 fixed_reason>"}
```

prompt + completion 不超过 14000 字符（约 4096 token）。

## 代码结构和从头复现

别人拿到「代码仓库 + 数据包 + 两个公开模型」就能从头跑。数据包 `python -m scripts.pack_data` → `dist/claim_critic_data.tar`（不进 git，约 58 MB），只含重跑也得不到同样结果的东西：Claude 标注、外部数据原件、评测集；`MANIFEST.json` 记每个文件的行数、字节数、SHA-256，解压后 `python -m scripts.pack_data --check` 核对。

| 数据包里的文件 | 内容 |
| --- | --- |
| `external_data/ragtruth/` | RAGTruth 原件（github.com/ParticleMedia/RAGTruth，MIT） |
| `external_data/ragtruth_pp/hallucination_spans.csv` | RAGTruth++ 重标 |
| `outputs/rt/ansgrpo/train_all.jsonl` | 831 个训练原文 |
| `outputs/rt/minicpm/grpo/train_ref.jsonl` | 2B 原回答（采样有随机性） |
| `outputs/rt/minicpm/s1/fix_*.jsonl` | Claude 改对 |
| `outputs/rt/core/label_raw.jsonl`、`sev_label_raw.jsonl` | Claude 新增标注（跑完才有） |
| `outputs/rt/minicpm/grpo300{,b,c}/{answers,final}.jsonl` | 验收集 + Claude 金标 |
| `outputs/rt/rm/rm1/{calib,test}.jsonl` | RAGTruth++ 校准和测试 |

模型：openbmb/MiniCPM5-2B（提交 f974000）、Qwen3.5-9B；9B 金标 critic 的 LoRA（rm1__rm1_gold_9b）只用于评测，另外给。

从头跑的顺序（本地 = 不需要 GPU；GPU = 单张 5090）：

| # | 步骤 | 命令 | 在哪 | 状态 |
| --- | --- | --- | --- | --- |
| 1 | 原文池 | `python -m ragtruth.ans_data split` | 本地 | 已有 |
| 2 | 原回答 | `python -m ragtruth.ans_data ref2 --model policy_v0 --sys_qa 1` | GPU | 已有（数据包里给了） |
| 3 | Claude 改对 | `python -m ragtruth.policy_sft s1_fix` | 本地，Claude | 已有 |
| 4 | Claude 标类型和满意度 | `python -m ragtruth.core_label run` | 本地，Claude | 提示词已写，脚本待写 |
| 5 | 核心集对齐和切分 | `python -m ragtruth.core_build` | 本地 | 待写（对齐逻辑在 `gen_seed_rt.candidates`） |
| 6 | 派生 D1–D4、D7–D9 | `python -m ragtruth.core_derive all` | 本地 | 待写 |
| 7 | 训改写模型 v0、RAG SFT、critic、满意度、拒答 | `train.sft` / `run_mcrm.sh` | GPU | 脚本已有 |
| 8 | 改写模型 v0 采样 → D5 负样本、sev 待标样本 | `python -m ragtruth.core_derive rm2` / `sevpool` | GPU | 待写 |
| 9 | Claude 标 sev 样本 | `python -m ragtruth.sev_label` | 本地，Claude | 已有 |
| 10 | 训 RM2、sev | `run_mcrm.sh` | GPU | 脚本已有 |
| 11 | 循环一 | `run_gensev.sh` → `run_harvest.sh` → `run_mcrm.sh` | GPU | 已有，要换模型名 |
| 12 | 循环二 | `run_r3full.sh`（`INIT=rag-sft-core CRITIC=...`） | GPU | 已有，要加 INIT 参数 |
| 13 | 评测 | `critic_cap3`、`pol9b_cmp` | GPU + 本地 | 已有 |

| 代码位置 | 内容 |
| --- | --- |
| `common/` | 样本格式、JSONL 读写、模型调用封装（OpenAI 兼容 / Anthropic / Claude CLI）、运行清单 |
| `ragtruth/ans_data.py`、`policy_sft.py`、`core_label.py` | 第一、二步：原文池、原回答、Claude 改对和标注 |
| `ragtruth/gen_seed_rt.py` | 核心集对齐、种子 SFT 行（要按本文改） |
| `ragtruth/synth.py`、`gen_target_data.py`、`critic_harvest.py` | 改写 prompt、采集、入库 |
| `ragtruth/rm_data.py`、`rm_eval.py`、`sev_rm.py`、`sev_label.py`、`sat_build.py`、`refuse_data.py`、`discrim.py` | 各打分模型的数据、提示词、校准和评测 |
| `ragtruth/critic_cap3.py`、`pol9b_cmp.py`、`mc_accept_gen.py`、`gold.py`、`accept_compare.py` | 评测 |
| `train/sft.py` | SFT 和单字母打分模型训练（`--mode coldstart`） |
| `train/gen_grpo.py`、`gen_reward.py` | 循环一 GRPO 和奖励 |
| `train/ans_grpo.py`、`ans_reward.py`、`prefix_share_llama.py` | 循环二 GRPO、奖励、训练侧 prompt 共享 |
| `scripts/autodl_*.py`、`remote.py` | AutoDL：抢卡、开机、释放、远程执行 |
| `scripts/remote/` | GPU 机上的流程脚本；按角色起步 `bootstrap_fs.sh`；多 LoRA vLLM 必须用 `serve_loras.sh` |
| `scripts/dump_prompts.py`、`pack_data.py` | 导出提示词、打数据包 |
| `configs/rt/` | 训练超参 |
| `tests/` | 单测（不需要 GPU） |

```bash
pip install -e ".[dev]"
PYTHONHASHSEED=0 python -m pytest -q        # 全部单测，不需要 GPU、不调用模型
```

远端（AutoDL，文件存储模式）：数据、模型、adapter 都在 `/root/autodl-fs`；登录信息文件（`autodl_web.json`、`autodl_token.txt`、`smtp.json`）不进仓库。`python -m scripts.autodl_extra --fs --roles <角色> --gpus 1 --gpu "RTX 5090"`，角色见 `scripts/remote/bootstrap_fs.sh`。

## 待确认

1. 2.2 新增标注：792 条回答、约 81 次 Claude 调用。
2. sev 单独标注：约 800 条、约 40 次 Claude 调用。
3. RAG GRPO prompt 池去掉 81 个留出原文（750 个）。
4. 更多样的回答用 RAGTruth 6 个模型在同一批原文上的回答；RAGTruth 人工标注只用来挑没有幻觉的回答，标签本身不进训练。
5. 拒答删段落用词重合门槛定标签，不再每条让 Claude 复判。
6. 旧 sev / RM3 标签不进新循环。
7. critic 训练集不放拒答句。
8. 数据包不进 GitHub，单独给人。
