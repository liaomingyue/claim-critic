# claim-critic：降低 RAG 幻觉的端到端方案

**逐句检测、强化学习与盲区闭环**

代码、数据和模型全部开源：[数据集](https://huggingface.co/datasets/liao11/claim-critic-data) · [模型（40 个 LoRA）](https://huggingface.co/liao11/claim-critic-models) · [发布与复现说明](docs/RELEASE.md)

**目标：** 让 RAG 模型的回答句句有原文依据。

**难在哪：**

1. **错误和正确的改写只差一线。** 最常见的幻觉不是凭空编造，而是对原文的轻微偏移：范围被放大、事情安到了别人头上、限定词丢了。它们和合理的概括、转述几乎长得一样，连人工标注都会大量漏掉：RAGTruth 里标为"无幻觉"的句子，抽查有 28% 其实有问题；我们的检测模型对"范围偏移"这一类，漏判率一度高达 76%。
2. **检测器一旦当奖励，就会被针对。** 用它做强化学习，生成模型会主动把错误挪到检测器看不见的地方，比如写长句、合并句子。离线评测更准的检测器，当奖励反而可能训出更差的模型。检测器面对的是一个会适应它的对手。
3. **补盲区的样本几乎造不出来，GRPO 也学不起来。** 一条有用的样本要同时满足三件事：是真错、读着自然、critic 拿不准。这三个目标互相拉扯：自然的错 critic 多半认得出，critic 认不出的又多半不是真错或者不自然。让生成器自由造错时，三者同时满足的只有约 1.4%；改成指定错误类型后，2048 条里也只有约 40 条，64 组提示里有 42 组一条都没有。一组样本全是 0 分，GRPO 就没有组内差异可学。就算造出来了也难验：直接奖励"骗过检测器"，造出的只有 24% 是真错；和检测器同等能力的验证模型，在盲区里一样判不准。（前后试过的十三种做法见[技术分享详细版](docs/tech_share_details.md#造难样本前后试过的十三种做法)）

**效果：** 在 MiniCPM5-2B 上，含明显错误的回答从 25.2% 降到 4.0%，含任意错误（包括细微错）的从 35.8% 降到 9.1%。同样的数据和流程换到 Qwen3.5-9B 上做对照：2B 加整套方案大幅好于只做 SFT 的 9B，和 9B 做完第一段 GRPO 在明显错上打平；Claude 独立判下来，细微错 2B 还更少。

**方案：** 三个部分，分别对应上面三个难点，串成一个闭环。

1. **逐句幻觉检测**：训练一个 critic，判断回答的每一句有没有原文依据。数据不用公开的人工标注，而是让模型自己写回答、由强模型逐句改对和标注，标签更干净，错误分布也和要检测的模型一致；
2. **句子级奖励的强化学习**：用 critic 的逐句判断当 GRPO 的奖励，惩罚只落在出错的那一句；critic 每升级一次，就从上一轮模型接着训。奖励不只有 critic：拒答模型、满意度模型和几条硬规则，专门堵住"全部拒答、照抄原文、答非所问"这些也能让幻觉变少的偷懒策略。衡量 critic 好坏，看它当奖励的效果，而不只看离线指标；
3. **盲区闭环**：持续按错误类型找出 critic 的盲区，生成器只造"critic 拿不准"的难样本，由更强的独立判断者（人，这里用 Claude 代替）核实后补进 critic，再用更好的 critic 接着训 RAG。

这不是某个单点算法，而是从数据构建、检测模型、奖励设计、RL 训练，到盲区分析、造样本、人工核实、评测和复现的一整套方案。

训练的全部模型都是 MiniCPM5-2B（下称 2B）+ LoRA。Qwen3.5-9B 金标 critic 只用于评测，不进奖励、不筛训练数据。Claude（claude-opus-5-5）代替人工，负责标注、核实和最终验收。

- 技术分享（几分钟看懂做了什么、有哪些发现）：[docs/tech_share.md](docs/tech_share.md)；详细版（表 1–9、论证、踩过的坑）：[docs/tech_share_details.md](docs/tech_share_details.md)
- 全部实验（按主题整理：目的、做法、结果、结论）：[docs/experiments.md](docs/experiments.md)
- 每个模型的训练数据和配置：[docs/MODELS.md](docs/MODELS.md)
- 奖励函数设计（RAG 的 R3、错误生成器的门槛 + 难度）：[docs/rewards.md](docs/rewards.md)
- 开源数据、模型与复现：[docs/RELEASE.md](docs/RELEASE.md)
- 提示词原文：[docs/prompts.md](docs/prompts.md)
- 金标规范：[docs/gold_standard.md](docs/gold_standard.md)

## 主要结论

训练方法上：

1. **GRPO 远比 SFT 有效。** 同一批题目，用 Claude 改对的回答做 SFT，每条回答的显然错句最多降到约 0.30（加轮数也不再降，过多还会过拟合）；GRPO 200 步降到 0.145。拒绝采样 SFT 几乎不降幻觉。
2. **奖励要落到句子上。** 只把惩罚加在出错那一句的 token 上，显然错句比整段一个分少 43%（−0.210 [−0.309, −0.120]）。
3. **critic 升级后，从上一轮 GRPO 模型接着训。** 比从 SFT 重新训少 5.0 个点含显然错 [−7.6, −2.3]。

和直觉相反的：

4. **静态指标更好的 critic，当奖励不一定更好。** 撞见了四次：AUC 更高的反而训出更差的 RAG，AUC 没变甚至变差的反而训得更好。和 RL 结果一致的是"被钻空子程度"，不是 AUC。
5. **让生成器自己挑错误类型，它会挑对 critic 最没用的。** 它偏爱模糊附加评价、添加事实这些 critic 本来就会判的错，自述的改法和实际机制一致率只有 2–27%。所以改成指定错误类型、但让它自己挑最合适的句子。
6. **知道改了什么，还是判不准是不是错。** sev 看得到原句和改写句，在 critic 拿不准的区域排序 AUC 也只有 0.67–0.71。定位不等于判断；同源的小模型和 critic 一起失明，所以核实必须交给更强的独立判断者。

小模型对上大模型（同样的数据、流程和超参，基座换成 Qwen3.5-9B）：

7. **2B 加整套方案，至少追平 9B 加第一段 GRPO。** 9B 评测器下两者打平（含明显错 4.0% vs 3.2%，含任意错 9.1% vs 9.6%），对只做 SFT 的 9B 则少 11.3 / 20.5 个点。Claude 对 100 对回答独立盲判：明显错打平，含任意错 2B 少 10 个点（22% vs 32%，临界显著）。方法本身在 9B 上同样有效，100 步就从 15.3% 降到 3.0%。
8. **评测器不能和奖励模型同源。** 9B 策略是拿 9B critic 训的，同底座的 9B 评测器在它的输出上和 Claude 的一致性只有 κ 0.14（2B 输出上 0.32），漏掉了大半细微错。关键对照要有独立裁判。

每一条的数据、解释和值得深挖的方向，见 [docs/tech_share_details.md](docs/tech_share_details.md)（「GRPO 和 SFT 比」「三个反常识的发现」「换成 9B」）。

## 现状

| 部分 | 状态 |
| --- | --- |
| RAG GRPO：句子级奖励 R3 + 分段接着训 | ✅ 稳定有效。含显然错 25.2%（RAG-SFT）→ 9.8%（S1）→ 6.6%（S2）→ **4.7%（S3-bf）**；含任意错 35.8% → 10.4% |
| **盲区驱动闭环：第一轮** | ✅ **有效**：补盲区的 critic（bfcl）接着训 RAG，含任意错 16.8% → 10.4%（对起点显著）；同步数对照（奖励模型不换）比较：两次独立运行含显然错都显著更低（−3.2、−2.0 个点），含任意错一次显著、一次临界 |
| 盲区驱动闭环：第二轮 | ⚪ 收益递减：含显然错 4.7% → 4.0%、含任意错 10.4% → 9.1%，仍在降但已不显著；**闭环到此停止**，转向错误体系 + 适合度模型 |
| critic 盲区分析：按 9 个细类统计 RAG 错误和 critic 漏判 | ✅ 已建立，每轮复用 |
| 定向造错 → 核实 → 补进 critic | ✅ 跑通：补盲区 581 条，critic 显然错 AUC +0.029；只补范围偏移，该类型 AUC +0.030 |
| 难样本合成：迭代式（GRPO ↔ 采样 ↔ SFT，大提示池温度 0） | ⚪ 接近饱和：过门槛的样本里 critic 拿不准的一直卡在约 40 条 / 2048；放宽 sev 门槛后由 Claude 核实，第二轮真错率 57% |
| 错误体系：细类、扩充方法、各类型样本的维护 | 🔄 初版（9 个细类 + 每类的盲区 / 造难能力数据），下一步重点 |
| 9B 对照：同一套方法换到 Qwen3.5-9B | ✅ 9B critic 静态显然错 AUC 0.948（2B 0.843）；9B SFT → GRPO 200 步含明显错 15.3% → 3.2%；2B 整套方案和它打平，Claude 口径细微错更少；2B 起点 + 9B 打分模型的 GRPO 🔄 进行中。多机训练见 [docs/distributed.md](docs/distributed.md) |
| 开源与复现 | ✅ 数据、40 个 LoRA、每个模型的训练数据和配置已整理；在全新的 GPU 服务器上只用发布内容独立复现过：不用 GPU 重算全部对比逐位一致，critic 重新打分结论全部成立，RAG 重训复现了核心对照的方向（明显错上仍显著，效应从 3.2 缩小到 2.0 个点）。见 [docs/RELEASE.md](docs/RELEASE.md) |

## 还缺什么、还没验证什么

1. **系统化的错误体系**：现在只有 9 个机制类。需要分层（粗类 → 机制 → 子类）、从真实错误里归纳子类、每个子类一张卡片（定义、正反例、适用条件、造法、核实要点、指标、样本库、版本），并用它来驱动评测、补样本配额、critic 回归门槛和生成器出题。尚未开始。
   这也是下一步要建**结构化错误体系**的原因：9 个粗类还不够。同一个粗类里，各种具体改法的难度和适用的句子差别很大，只指定粗类，生成器仍然是在大海里捞针。把类型细化到子类，并且只在适合这个子类的句子上出题，才有可能把"三个条件同时满足"从百分之一二的偶然事件，变成每道题都有较大机会命中的条件生成任务，GRPO 才有组内差异可学。这是我们的判断，还没有实验验证。
2. **其他任务上的验证**：只在 RAGTruth 的问答和摘要、2B 模型上验证过。智能搜索（多来源、噪声、来源冲突）和 Agent（多步工具调用轨迹）还没做；critic 的输入形式和错误体系都需要扩展。
3. **适合度模型**：指定错误类型后，生成器只有约 31% 遵守，原因是很多句子不适合这类错（没有数字却要造数字错等）。需要一个模型在大批句子里挑出适合改写的（句子，类型）组合，只在合适的句子上改写。第一版用规则，之后用生成成功率数据训模型。
4. 现有造样本方式接近饱和；高重合、只改一处的显然错盲区还没覆盖。
5. 人工（Claude）核实成本高；Claude 和真人的一致性没测；9B 评测器在大的变化上和 Claude 一致，但在 RL 训练后的输出上只是中等可靠，比例偏低，几个点的小差距方向没有保证，评测拿同源 critic 训出的模型时更不可靠（9B 对照里 κ 只有 0.14）；没有多种子重复，而独立重跑一次训练，最终结果能差约 2 个点。

## 闭环

```
            ┌──────────────────────────────────────────────────────────────────────┐
            ▼                                                                      │
 ① RAG GRPO（critic 当奖励，R3 句子级奖励；从上一轮 GRPO 模型接着训）                │
            │ RAG 输出                                                             │
            ▼                                                                      │
 ② 分析：RAG 输出 + critic 盲区，按错误体系（9 细类）统计                              │
            │ 找「RAG 常犯 且 critic 常漏」的类型                                    │
            ▼                                                                      │
 ③ 定向造错：错误生成器按类型改写干净句 → 门槛（sev 真错、RM2 自然）                   │
            │ → 难度筛选（critic 0.1–0.5）；生成器本身迭代（GRPO → 采样 → SFT）       │
            ▼                                                                      │
 ④ 核实：Claude（代替人工）判真错 / 类型 / 自然；只有这一步需要人                      │
            │                                                                      │
            ▼                                                                      │
 ⑤ critic 重训（核心集 + 历轮核实样本）→ 按类型复查 critic ──────────────────────────┘
```

| 步 | 做什么 | 入口 | 判定标准 |
| --- | --- | --- | --- |
| ① | R3 奖励 GRPO：起点是上一轮 GRPO 的检查点，不从 SFT / base 重来 | `scripts/remote/run_loop2.sh`、`run_loop2cont.sh`（`START=` 上一轮 adapter、`CRITIC=` 新 critic） | 9B 金标：每条回答的显然错句数、含显然错比例；三套验收原文，配对 bootstrap |
| ② | Claude 金标验收集上，按细类统计 critic 召回；RAG 输出按细类统计错误分布 | `ragtruth/critic_cap3.py --by_type` | 两侧都差的类型优先 |
| ③ | 生成器 v2：输入原文 + 回答 + 要求的细类，输出改哪句和改写句。GRPO 奖励：门槛（sev P(有错) ≥ 0.9、RM2 自然 ≥ 0.5）+ critic 难度（0.3–0.5 最高，≥ 0.9 为 0） | `ragtruth/gen_v2.py`、`train/gen_reward_v2.py`、`ragtruth/gen_v2_grpo.py`、`ragtruth/gen_pool_rag.py`、`scripts/remote/run_gen_*.sh` | 64 个留出提示 × 32 试采：过门槛样本里 critic < 0.5 的比例、0.1–0.5 条数、类型一致率 |
| ④ | 难区样本交 Claude：真错 / 实际机制 / 是否等于要求类型 / 自然 | `ragtruth/gen_v2_label.py`、`ragtruth/scope_critic.py` | 真错 + 类型一致 + 自然 |
| ⑤ | 核实样本（配或不配同位置原句）加进 critic 训练集重训 | `ragtruth/critic_harvest.py`、`ragtruth/scope_critic.py build`、`scripts/remote/run_mcrm.sh` | cap3：显然错 AUC / 误报 10% 召回（主口径）+ 任意错 + 按类型 |

### ① RAG GRPO：用 critic 当奖励训 RAG

- 奖励 R3：
  - 整段层：去掉同组底噪，按错句计数，K = max(2, 0.5 × 参考句数)；
  - 句子层优势：只打到出错的那一句，β = 0.6；后一句的负值按 γ = 0.3 传给前一句；
  - 满意度只罚不奖（最多扣 0.4），拒答 0.4 × P(可拒答)，外加硬规则。
  - 公式见下文「奖励公式」。
- 分段接着训：每轮 critic 更新后，从上一轮 GRPO 检查点接着训 100 步（lr 5e-6，16 prompt × 16），KL 相对新起点，参考回答由新起点重采。
- 结果（9B 金标 critic 打分，三套验收原文上 595 条配对回答 / 299 个原文；`results/loop2_cont6_cmp.json` 等）：

| 模型 | 含显然错 | 显然错句 / 条 | 含任意错 | 拒答 |
| --- | --- | --- | --- | --- |
| RAG-SFT（rag-sft-core） | 25.2% | 0.358 | 35.8% | 27.6% |
| RAG-S1 第一段：critic rm1__core，200 步 | 9.8% | 0.106 | 21.7% | 6.0% |
| 对照 RAG-S1x：SFT 起点一次训完，critic r1，200 步 | 11.6% | 0.128 | 21.3% | 5.9% |
| **RAG-S2 第二段：S1 起点，critic r1，100 步** | **6.6%** | **0.072** | 16.8% | 3.5% |
| RAG-S3-r2 第三段：S2 起点，critic r2，100 步 | 8.1% | 0.081 | 14.6% | 1.5% |
| RAG-S3-r3 第三段：S2 起点，critic r3（长句增强），100 步 | 5.4% | 0.056 | 14.0% | 2.2% |
| 对照 RAG-S3-r1：S2 起点，critic 不换（r1），100 步 | 7.7% | 0.081 | 14.3% | 2.2% |
| **RAG-S3-bf 第一轮闭环：S2 起点，补盲后的 critic（bfcl），100 步** | **4.7%** | **0.047** | 10.4% | 1.9% |
| RAG-S4-rd2 第二轮闭环：S3-bf 起点，critic rd2，100 步 | 4.0% | 0.040 | 9.1% | 1.3% |

同一个新 critic（r1）下，分段接着训比一次训完少 5.0 个点含显然错 [−7.6, −2.3]（90% 区间）。同样多训 100 步，换成补盲后的 critic（S3-bf）比奖励不换（S3-r1）含显然错少 3.2 个点 [−5.3, −1.2]、含任意错少 4.0 个点 [−6.6, −1.5]。复现：独立重跑一次两组，新 critic 组含显然错仍显著低于对照（−2.0 [−3.6, −0.5]），含任意错为临界（−2.4 [−4.7, +0.0]）；对照组这次自身也有改善（含显然错 5.9%），说明同一设置重跑能差约 2 个点。第二轮（S4-rd2）对 S3-bf 的下降不再显著。

### ② 分析：critic 在哪儿看不见

每一轮 RAG 训完，在 Claude 金标验收集上按 9 个细类（见下文「错误体系」）分别算 critic 的召回，同时统计 RAG 输出里各类错误出现多少。两边都差的类型，就是下一轮要补的目标。

- 补盲之前，金标 critic 漏掉的 339 个错误里，范围 / 确定性偏移占 35%（这一类的漏判率高达 76%），其次是近义偏移 15%、张冠李戴 13%。而早期生成器造出的真错里，最多的是模糊附加语和添加具体事实，正好是 critic 本来就会判的：**合成数据和 critic 的盲区是错位的**。
- critic 漏掉的显然错，大多不是"很确定地认为没错"，而是"有点怀疑但没过阈值"：漏掉的里面只有约五分之一 P < 0.1，其余都在 0.1–0.5。所以后面选样本时取 critic 0.1–0.5 这一段。
- 第一轮补盲之后，再看还漏的显然错：范围偏移漏 65%、近义偏移 64%、张冠李戴 45%；和原文字面重合度高的句子漏得最多（60%）。读下来几乎都是"整句照抄原文、只改一处"，比如事件或对象错配、角色关系颠倒、限定词丢失。这类错用现在的方式很难造出来，也是闭环停下的原因之一。

入口：`ragtruth/critic_cap3.py --by_type`（按类型的召回和 AUC）、`scripts/analysis/clear_miss_an.py`（漏判分析）。细节见 [docs/experiments.md](docs/experiments.md) §9。

### ③ 定向造难样本

- 生成器只在干净句上造错：
  - 问答对：RAG GRPO 最好版本（cont 第 100 步）在 RAGTruth 训练原文上生成回答。温度 1.0、每个原文 8 条，多样性来自这里。
  - 干净句：2B critic 逐句打分，P(有幻觉) < 0.1 的句子。
  - 提示：每个回答 × 9 种类型，每个提示温度 0 只采 1 条。
- 迭代：
  1. GRPO 50 步（critic 难度奖励，提示按类型加权）；
  2. 在大提示池上采样，挑过门槛、critic 0.1–0.5、sev 判要求类型 ≥ 0.5 的样本；
  3. 难样本 + 原 SFT 数据做 SFT；
  4. 试采。
  - 脚本：`scripts/remote/run_gen_b.sh` → `run_gen_c.sh` → `run_gen_d.sh`。
- 打分模型（都是 2B 单字母选项 + 全局温度校准）：
  - sev v3c0：要求的那种·显然 / 细微 / 别的错 / 没错；
  - RM2 v3b：自然度；
  - critic rm1__core_r2。

### ④ 核实：人只出现在这一步

入库前由 Claude 核实，Claude 代替人工；训练中、评测中都不调 Claude。

- 只补范围偏移：核实组该类型 AUC 显著 +0.030，不核实组 +0.007 不显著。
- 9B 生成器的样本，即使是 Claude 判为真错的 811 条，补进 critic 也没有提升（−0.010）：决定 critic 能否变好的，是「critic 拿不准 + 类型对口」，不只是「真错」。

### ⑤ critic 重训

把核实过的样本加进 critic 的训练集，从基座重新训一个 LoRA（超参不变，`configs/rt/rm_mc.yaml`），再在同一个验收集上按总体和按类型复查。

| critic | 训练数据 | 结果（Claude 金标验收集） |
| --- | --- | --- |
| rm1__core | 核心集 4604 行 | 显然错 AUC 0.843 |
| rm1__core_r2 | + critic 拿不准、Claude 核实为真错的 581 条（配同位置原句） | 0.872，+0.029 [+0.015, +0.044] |
| rm1__core_r2_bfcl（第一轮） | r2 + 四类盲区补样本，Claude 核实后 558 条（只加错句） | 对不核实的对照组（bfnc，同样条数）：显然错 +0.020 [+0.008, +0.034]、任意错 +0.010 [+0.003, +0.019] |
| rm1__core_rd2（第二轮） | bfcl + 第二轮核实的 799 条 | 静态指标反而变差：任意错 AUC −0.019 [−0.030, −0.008]；整体更"多疑"，没错的句子 P < 0.1 的比例从 54% 降到 40% |

两点经验：
- **加什么比加多少重要。** 随机加几千条合成样本没用，加 Claude 确认的真错也没用，只有"critic 拿不准 + 类型对口 + 核实过"的样本才有用（技术分享详细版表 3）。
- **静态指标和最终效果不完全一致。** rd2 的 AUC 下降了，但拿它当奖励训出来的 RAG（S4-rd2）错误还在小幅下降（含显然错 4.7% → 4.0%，不显著）。判断一个 critic 好不好，最终要看它当奖励时 RAG 变成什么样。

入口：`ragtruth/critic_harvest.py`、`ragtruth/blind_fill.py build`、`scripts/remote/run_mcrm.sh`。每个 critic 的完整训练数据见 [docs/MODELS.md](docs/MODELS.md)。

## 错误体系（初版）

9 个细类，生成器提示、sev、Claude 标注和 critic 按类型评测共用同一套名字（`ragtruth/gen_v2.MECH`、`ragtruth/core_label.py`）。每一类需要维护：定义、critic 召回、RAG 出错率、造难能力、扩充方法，以及已核实的样本库。

| 细类 | critic 漏判占比* | 造难能力（难样本数 / 难样本里还是该类型） | 扩充方法 / 备注 |
| --- | --- | --- | --- |
| certainty_or_scope_shift 范围 / 确定性偏移 | 35%（该类漏判率 76%） | 最多（191），类型保持最好 | critic 最大盲区；Claude 标注多为「细微」，主口径看不见，要看任意错口径 |
| near_synonym_shift 近义偏移 | 15% | 中 | 改一两个词，真假依赖原文 |
| wrong_attribution 张冠李戴 | 13% | 中 | sev 判类型提升最大（0.53 → 0.81） |
| contradicts_source 直接矛盾 | 13% | 难：造难后多变成别的类型 | 是「结果」不是「手法」，sev 放宽为「真错即可」 |
| merged_across_passages 跨段拼接 | 11% | 中 | 生成器产出偏少 |
| unsupported_inference 无依据推断 | — | 造难后常变类型 | |
| number_or_date 数字日期 | — | 难样本最少，但多为真错 | |
| vague_evaluation_or_addon 附加评价 | 1% | 易造，但 critic 本来就会判 | 生成器天然偏好，需降权 |
| added_specific_fact 添加具体事实 | 1% | 造难后常变类型 | critic 最会判 |

\* 金标 critic 漏掉的 339 个错误里各类型的占比（见 [docs/experiments.md](docs/experiments.md) 的 critic 盲区分析）。

下一步：
- 把每一类做成一张卡片：定义、正反例、扩充提示、门槛、样本库、各轮指标；
- 加一个「适合度」模型：给（问题，回答，原文）打分，判断最适合造哪类错，替代现在的「每个回答 × 9 类」。

## 数据与模型

整个项目只有一个起始数据集，即**核心集**：2B 在 831 个 RAGTruth 训练原文上写的回答（807 条），由 Claude 逐句改对并标注类型、严重程度和满意度。其余数据集都从核心集派生，81 个原文留出，只用于评测。

| # | 数据集 | 用途 |
| --- | --- | --- |
| D1 | RAG SFT（改对后的回答） | rag-sft-core |
| D2 | RAG GRPO 原文池 750 | 循环二 |
| D3 | 种子错误对 1472 / 留出 155 | 错误生成器 SFT |
| D4 | 改写提示池（核心集回答 657 × 9 细类） | 生成器 GRPO / 试采 |
| D5 | RM2 自然度（Claude「自然」标签） | 门槛 |
| D6 | sev（生成器真实输出 + Claude 标注） | 门槛 |
| D7 / D8 | 满意度 / 拒答 | 循环二护栏 |
| D9 | critic：真错句 + 同位置改对句 + 没改的句子 | rm1__core；之后 + 核实样本 → r1 / r2 / r3 |
| — | 大提示池：RAG GRPO 模型回答 × 干净句 × 9 类（690 原文） | 难样本合成（`gen_pool_rag`） |

| 模型 | 名字（adapters/） |
| --- | --- |
| RAG | RAG-SFT（rag-sft-core）→ RAG-S1 第 1 段强化（…_coreonly）→ RAG-S2 第 2 段（…_cont）→ RAG-S3-r2 / -r3 / -bf（第 3 段的三个平行尝试，…_cont2 / 3 / 4）+ 对照 RAG-S3-r1（…_cont5）→ **RAG-S4-rd2**（第二轮闭环，…_cont6，最终版）。每个版本的起点、奖励和训练数据见 [docs/MODELS.md](docs/MODELS.md) |
| critic | rm1__core → rm1__core_r1（+ 循环一采集）→ rm1__core_r2（+ Claude 核实的盲区 581）→ r3（+ 长句增强）；闭环：r2 → **rm1__core_r2_bfcl**（第一轮，多类型补盲 + Claude 核实；对照 bfnc 不核实）→ rm1__core_rd2（第二轮） |
| 错误生成器 | gen-core-mc（v1）→ gen-v2-mc（v2：类型条件）→ gen-v2h / v2h2（难样本 SFT）→ g1（GRPO）→ v2h3 |
| 打分 | sev__v3c0 / sev__v3d、rm2__v3b / rm2__v3d（v3d 加了第一轮盲区的 Claude 标注）、sat_pair__core、refuse__core |
| 评测 | rm1__rm1_gold_9b（9B，只评测） |

## 评测

三个阶段各有固定评测集，都和训练原文不重叠；评测中不调 Claude。

- **RAG 生成**：三套验收原文（a：100 条，T 1.0；b：100 条，T 0.7；c：132 原文 × 3 种子，T 0.7；共约 596 条）。每个策略在同样的原文和种子上重新生成，9B 金标 critic 逐句打分（p ≥ 0.9 记显然错），按原文聚类 bootstrap（`ragtruth/pol9b_cmp.py`）。9B 评测器和 Claude 的一致性（六组回答，3,839 句）：训练前后十几个点的大变化两者同向、幅度相近；但在 RL 训练后的输出上只是中等可靠（2B GRPO 三套合并：显然错 AUC 0.822、回答级 κ 0.43），报的比例约为 Claude 的六成，几个点的小差距方向没有保证。所以文中比例都应看作 9B 口径下的相对变化（`scripts/analysis/eval9b_agreement.py`，[experiments.md §1.5](docs/experiments.md)）。
- **critic**：cap3（三套验收回答的 Claude 金标，去掉拒答，2061 句，其中显然错 179），`ragtruth/critic_cap3.py`。
  - 主口径：显然错 vs 没错的 AUC、误报 10% 时的召回；
  - 附加：任意错（含细微）口径、按细类（`--by_type`）、RAGTruth++ 2306 句。
- **错误合成**：64 个留出提示 × 32 试采（`gen_v2_grpo probe`），看各道门槛的通过率、过门槛样本的 critic 分布、全 0 组比例；训练中每 10 步看门槛通过率（`scripts/gate_stats.py`）。

## 奖励公式

完整说明（硬规则、每个参数的取值和理由、做过的变体）见 [docs/rewards.md](docs/rewards.md)。

**RAG GRPO（R3，`train/ans_reward.py`）**。记号：p_i = critic 对第 i 句的 P(错)；b、σ = 同组所有回答所有句子的 P 均值和标准差（σ 下限 0.1）；e_i = max(0, p_i − b)。

```
H = 1 − 0.5 · (1 − Π_i (1 − e_i)) − 0.5 · min(1, Σ_i e_i / K),  K = max(2, 0.5 · n_ref)
r = 0（没过硬规则）| 0.4 · P(可拒答)（拒答）| max(0, H − 0.8 · max(0, 0.5 − P_sat))
token 优势 = 整段组内归一化优势 + 0.6 · (−e_i / σ)（只加在第 i 句的 token 上；后一句 × 0.3 传给前一句）
```

**错误生成器 GRPO（`train/gen_reward_v2.py`，`--v2_shape hard --hard_shape tri --hard_base 0`）**：

```
门槛：硬规则 ∧ sev P(有错) ≥ 0.9 ∧ RM2 P(自然) ≥ 0.5，没过 0 分
h(p) = 0（p < 0.1）| (p − 0.1)/0.2（0.1–0.3）| 1（0.3–0.5）| (0.9 − p)/0.4（0.5–0.9）| 0（≥ 0.9）
r = h(critic P(有幻觉))
```

## 复现与运行

```bash
pip install -r requirements.txt && pip install -e .
python -m scripts.demo_tables               # 不需要 GPU：从 results/ 重出技术分享里的表 1–9
PYTHONHASHSEED=0 python -m pytest -q        # 单测，不需要 GPU、不调用模型
pip install -r requirements-gpu.txt         # GPU 端（训练 / vLLM），版本同训练机
```

三个复现层级（L0 只看结果 / L1 从逐句打分重算 / L2 完整闭环）、公开的数据和模型（Hugging Face：[数据集](https://huggingface.co/datasets/liao11/claim-critic-data)、[模型](https://huggingface.co/liao11/claim-critic-models)）及许可说明，见 [docs/RELEASE.md](docs/RELEASE.md)。代码许可证：Apache-2.0。每个模型（40 个 LoRA）的训练数据和训练配置见 [docs/MODELS.md](docs/MODELS.md)。

- 全部运行用 `PYTHONHASHSEED=0` + 固定字符串种子，数据构建重跑逐位一致。用 vLLM 的生成和 GRPO 训练做不到逐位一致（GPU 批次和浮点累加顺序的细小差异会被逐步放大），只能保证趋势一致；我们在新服务器上重跑验证过：重新打分同一句平均差 0.005，重训 RAG 数字差约 2 个点，结论方向不变。
- 数据和模型不进 git，发布在 Hugging Face：`python -m scripts.fetch_release` 下载并校验 SHA-256（见 [docs/RELEASE.md](docs/RELEASE.md)）。
- GPU 端（AutoDL，单张 RTX 5090）：`python -m scripts.autodl_extra --fs --roles <角色> --gpus 1 --gpu "RTX 5090"`，角色见 `scripts/remote/bootstrap_fs.sh`。
  - 多 LoRA vLLM 一律用 `scripts/remote/serve_loras.sh` 启动。
  - 9B 的 GRPO 一张卡放不下：拆成一台训练机 + 一台或多台推理机（采样和打分分摊到各推理机，每步 LoRA 后台推给每一台），见 [docs/distributed.md](docs/distributed.md)。
  - 训练、试采只读写本地盘 `/root/autodl-tmp`，共享盘 `/root/autodl-fs` 由 `scripts/remote/bg_sync.sh` 在后台同步。
  - 登录信息文件（`autodl_web.json`、`autodl_token.txt`、`smtp.json`）不进仓库。

| 代码位置 | 内容 |
| --- | --- |
| `common/` | 样本格式、JSONL、模型调用封装（OpenAI 兼容 / Anthropic）、运行清单 |
| `ragtruth/core*.py`、`ans_data.py`、`policy_sft.py` | 核心集：原文池、回答、Claude 改对和标注、派生数据 |
| `ragtruth/gen_v2*.py`、`gen_pool_rag.py`、`sev_gen.py`、`rm2_v3.py` | 错误生成器 v2、大提示池、sev / RM2 |
| `ragtruth/critic_harvest.py`、`scope_critic.py`、`critic_sevaug.py` | 核实样本入库、定向补 critic |
| `ragtruth/critic_cap3.py`、`pol9b_cmp.py`、`mc_accept_gen.py`、`gold.py` | 评测 |
| `train/sft.py` | SFT 和单字母打分模型 |
| `train/ans_grpo.py`、`ans_reward.py` | RAG GRPO + R3 |
| `train/gen_grpo.py`、`gen_reward_v2.py` | 错误生成器 GRPO + 难度奖励 |
| `train/prefix_share_llama.py` | 训练侧 prompt 前缀共享 |
| `scripts/remote/` | GPU 端流程脚本 |
| `configs/` | 训练超参、模型配置（`configs/models/`） |
