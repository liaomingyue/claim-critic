# claim-critic：RAG 逐句幻觉 critic 与「盲区驱动」的迭代优化闭环

对 RAG 回答做逐句幻觉检测的 critic（判每句话有没有原文依据），并用它驱动一个闭环：

- critic 当奖励训练 RAG 回答模型；
- 再按错误类型分析 RAG 输出和 critic 的盲区；
- 针对盲区合成难样本、核实后补进 critic；
- 用更好的 critic 从上一轮的 RAG 模型接着训。

训练的全部模型都是 MiniCPM5-2B（下称 2B）+ LoRA。Qwen3.5-9B 金标 critic 只用于评测，不进奖励、不筛训练数据。Claude（claude-opus-5-5）代替人工，负责标注、核实和最终验收。

- 技术分享（做了什么、哪些有效、踩过的坑）：[docs/tech_share.md](docs/tech_share.md)
- 完整实验记录：[docs/experiment_log.md](docs/experiment_log.md)
- 结果汇总（截至 11-04）：[docs/results.md](docs/results.md)
- 提示词原文：[docs/prompts.md](docs/prompts.md)
- 金标规范：[docs/gold_standard.md](docs/gold_standard.md)

## 现状（2026-11-09）

| 部分 | 状态 |
| --- | --- |
| RAG GRPO：句子级奖励 R3 + 分段接着训 | ✅ 稳定有效。含显然错 25.2%（RAG-SFT）→ 9.8%（S1）→ 6.6%（S2）→ **4.7%（S3-bf）**；含任意错 35.8% → 10.4% |
| **盲区驱动闭环：第一轮** | ✅ **有效**：补盲区的 critic（bfcl）接着训 RAG，含任意错 16.8% → 10.4%（对起点显著）；对照（奖励模型不换、同样多训 100 步）无改善，提升来自新 critic |
| 盲区驱动闭环：第二轮 | ⚪ 收益递减：含显然错 4.7% → 4.0%、含任意错 10.4% → 9.1%，仍在降但已不显著；**闭环到此停止**，转向错误体系 + 适合度模型 |
| critic 盲区分析：按 9 个细类统计 RAG 错误和 critic 漏判 | ✅ 已建立，每轮复用 |
| 定向造错 → 核实 → 补进 critic | ✅ 跑通：补盲区 581 条，critic 显然错 AUC +0.029；只补范围偏移，该类型 AUC +0.030 |
| 难样本合成：迭代式（GRPO ↔ 采样 ↔ SFT，大提示池温度 0） | 🔄 进行中。难样本（critic 0.1–0.5）产出还低，是当前主攻点 |
| 错误体系：细类、扩充方法、各类型样本的维护 | 🔄 初版（9 个细类 + 每类的盲区 / 造难能力数据），下一步重点 |

## 还缺什么、还没验证什么

1. **系统化的错误体系**：现在只有 9 个机制类。需要分层（粗类 → 机制 → 子类）、从真实错误里归纳子类、每个子类一张卡片（定义、正反例、适用条件、造法、核实要点、指标、样本库、版本），并用它来驱动评测、补样本配额、critic 回归门槛和生成器出题。尚未开始。
2. **其他任务上的验证**：只在 RAGTruth 的问答和摘要、2B 模型上验证过。智能搜索（多来源、噪声、来源冲突）和 Agent（多步工具调用轨迹）还没做；critic 的输入形式和错误体系都需要扩展。
3. **适合度模型**：指定错误类型后，生成器只有约 31% 遵守，原因是很多句子不适合这类错（没有数字却要造数字错等）。需要一个模型在大批句子里挑出适合改写的（句子，类型）组合，只在合适的句子上改写。第一版用规则，之后用生成成功率数据训模型。
4. 现有造样本方式接近饱和；高重合、只改一处的显然错盲区还没覆盖。
5. 人工（Claude）核实成本高；Claude 和真人的一致性、9B 评测器的同源偏差都没测；没有多种子重复。
6. 已完成：「接着训」对照证明提升来自新 critic；第二轮闭环持平，说明现有造样本方式到头。

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

### ① RAG GRPO（循环二）

- 奖励 R3：
  - 整段层：去掉同组底噪，按错句计数，K = max(2, 0.5 × 参考句数)；
  - 句子层优势：只打到出错的那一句，β = 0.6；后一句的负值按 γ = 0.3 传给前一句；
  - 满意度只罚不奖（最多扣 0.4），拒答 0.4 × P(可拒答)，外加硬规则。
  - 公式见下文「奖励公式」。
- 分段接着训：每轮 critic 更新后，从上一轮 GRPO 检查点接着训 100 步（lr 5e-6，16 prompt × 16），KL 相对新起点，参考回答由新起点重采。
- 结果（9B 金标，三套验收 595 条配对回答 / 299 原文，`outputs/rt/core/loop2_all_cmp.json`）：

| 模型 | 含显然错 | 显然错句 / 条 | 含任意错 | 拒答 |
| --- | --- | --- | --- | --- |
| RAG SFT（rag-sft-core） | 25.2% | 0.358 | 35.8% | 27.6% |
| 第一段：critic rm1__core，200 步 | 9.7% | 0.106 | 21.7% | 6.1% |
| 一次训完：SFT 起点，critic r1，200 步 | 11.6% | 0.128 | 21.3% | 5.9% |
| **第二段：第一段起点，critic r1，100 步（cont）** | **6.6%** | **0.072** | 16.8% | 3.5% |
| 第三段：cont 起点，critic r2，100 步 | 8.1% | 0.081 | 14.6% | 1.5% |
| 第三段：cont 起点，critic r3（长句增强），100 步 | 5.4% | 0.056 | 14.0% | 2.2% |

同一个新 critic（r1）下，分段接着训比一次训完少 5.0 个点含显然错 [−7.6, −2.3]（90% 区间）。

### ③ 难样本合成（错误合成）

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

### ④ 人只出现在这里

入库前由 Claude 核实，Claude 代替人工；训练中、评测中都不调 Claude。

- 只补范围偏移：核实组该类型 AUC 显著 +0.030，不核实组 +0.007 不显著。
- 9B 生成器的样本，即使是 Claude 判为真错的 811 条，补进 critic 也没有提升（−0.010）：决定 critic 能否变好的，是「critic 拿不准 + 类型对口」，不只是「真错」。

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

\* 金标 critic 漏掉的 339 个错误里各类型的占比（`docs/experiment_log.md` 11-08「critic 盲区形态」）。

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
| RAG | RAG-SFT（rag-sft-core）→ RAG-S1 第 1 段强化（…_coreonly）→ RAG-S2 第 2 段（…_cont）→ RAG-S3-r2 / -r3 / **-bf**（第 3 段的三个平行尝试，…_cont2 / 3 / 4；-bf 当前最好）。命名对照见实验日志 11-09「RAG 模型统一命名」 |
| critic | rm1__core → rm1__core_r1（+ 循环一采集）→ rm1__core_r2（+ Claude 核实的盲区 581）→ r3（+ 长句增强） |
| 错误生成器 | gen-core-mc（v1）→ gen-v2-mc（v2：类型条件）→ gen-v2h / v2h2（难样本 SFT）→ g1（GRPO）→ v2h3 |
| 打分 | sev__v3c0、rm2__v3b、sat_pair__core、refuse__core |
| 评测 | rm1__rm1_gold_9b（9B，只评测） |

## 评测

三个阶段各有固定评测集，都和训练原文不重叠；评测中不调 Claude。

- **RAG 生成**：三套验收原文（a：100 条，T 1.0；b：100 条，T 0.7；c：132 原文 × 3 种子，T 0.7；共约 596 条）。每个策略在同样的原文和种子上重新生成，9B 金标 critic 逐句打分（p ≥ 0.9 记显然错），按原文聚类 bootstrap（`ragtruth/pol9b_cmp.py`）。
- **critic**：cap3（三套验收回答的 Claude 金标，去掉拒答，2061 句，其中显然错 179），`ragtruth/critic_cap3.py`。
  - 主口径：显然错 vs 没错的 AUC、误报 10% 时的召回；
  - 附加：任意错（含细微）口径、按细类（`--by_type`）、RAGTruth++ 2306 句。
- **错误合成**：64 个留出提示 × 32 试采（`gen_v2_grpo probe`），看各道门槛的通过率、过门槛样本的 critic 分布、全 0 组比例；训练中每 10 步看门槛通过率（`scripts/gate_stats.py`）。

## 奖励公式

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

不设保底：保底会让明显错比没过门槛更划算，生成器会往 ≥ 0.9 走。不单调奖励「骗过 critic」：完全骗过的样本大多不是真错。

## 复现与运行

```bash
pip install -r requirements.txt && pip install -e .
python -m scripts.demo_tables               # 不需要 GPU：从 results/ 重出技术分享里的表 1–6
PYTHONHASHSEED=0 python -m pytest -q        # 单测，不需要 GPU、不调用模型
pip install -r requirements-gpu.txt         # GPU 端（训练 / vLLM），版本同训练机
```

三个复现层级（L0 只看结果 / L1 从打分重算 / L2 完整闭环）见 [docs/tech_share.md](docs/tech_share.md) 第 6 节。

- 全部运行用 `PYTHONHASHSEED=0` + 固定字符串种子，重跑得到同样的数据。
- 数据包（Claude 标注、外部数据原件、评测集）不进 git：`python -m scripts.pack_data` → `dist/claim_critic_data.tar`，附 `MANIFEST.json`（SHA-256）。
- GPU 端（AutoDL，单张 RTX 5090）：`python -m scripts.autodl_extra --fs --roles <角色> --gpus 1 --gpu "RTX 5090"`，角色见 `scripts/remote/bootstrap_fs.sh`。
  - 多 LoRA vLLM 一律用 `scripts/remote/serve_loras.sh` 启动。
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
