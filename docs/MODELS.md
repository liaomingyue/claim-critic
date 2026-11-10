# 模型清单：每个模型的训练数据和训练配置

本页由 `python -m scripts.model_cards` 从 `configs/release_models.yaml` 和发布数据里的 `meta.json` / `eval/report.json` 生成，不要手改。
发布包和复现方法见 [RELEASE.md](RELEASE.md)。所有模型都只发 LoRA。「对应结论」里的表号指 [tech_share_details.md](tech_share_details.md) 的表 1–9。

共同约定：
- 打分模型（critic / sev / RM2 / 满意度 / 拒答）是"单字母选项"格式：模型只输出一个字母，取 logprob 当概率，再用校准集拟合一个全局温度；
- 所有打分模型共用一份超参（`configs/rt/rm_mc.yaml`），区别只在训练数据；
- 种子固定（seed 0、`PYTHONHASHSEED=0`）；RAG 和生成器的 GRPO 用 vLLM 采样，有少量非确定性。

## 总表

| 模型 | 类别 | 训练方式 | 起点 / 训练数据 | 对应结论 |
|---|---|---|---|---|
| rm1__core | critic | 选项 SFT | `outputs/rt/rm/rm1__core/` 4604 行 | 表 3（核心集底座）；表 2（RAG-S1 的奖励） |
| rm1__core_r1 | critic | 选项 SFT | `outputs/rt/rm/rm1__core_r1/` 5704 行 | 表 2（RAG-S2、RAG-S1x 的奖励）；表 8（对照组 RAG-S3-r1 的奖励） |
| rm1__core_r2 | critic | 选项 SFT | `outputs/rt/rm/rm1__core_r2/` 5766 行 | 表 3 最后一行（critic 拿不准的真错 581）；表 4、表 7 的底座；表 2（RAG-S3-r2 的奖励） |
| rm1__core_r3 | critic | 选项 SFT | `outputs/rt/rm/rm1__core_r3/` 7766 行 | 表 2（RAG-S3-r3 的奖励） |
| rm1__core_r2_sccl | critic | 选项 SFT | `outputs/rt/rm/rm1__core_r2_sccl/` 6336 行 | 表 4（单类型补盲，核实组） |
| rm1__core_r2_scnc | critic | 选项 SFT | `outputs/rt/rm/rm1__core_r2_scnc/` 6336 行 | 表 4（不核实组） |
| rm1__core_r2_sccl_no | critic | 选项 SFT | `outputs/rt/rm/rm1__core_r2_sccl_no/` 6051 行 | 表 4（配不配原句） |
| rm1__core_r2_bfcl | critic | 选项 SFT | `outputs/rt/rm/rm1__core_r2_bfcl/` 6324 行 | 表 7、表 8（RAG-S3-bf 的奖励） |
| rm1__core_r2_bfnc | critic | 选项 SFT | `outputs/rt/rm/rm1__core_r2_bfnc/` 6324 行 | 表 7（核实 vs 不核实） |
| rm1__core_rd2 | critic | 选项 SFT | `outputs/rt/rm/rm1__core_rd2/` 7123 行 | 表 9（静态评测；RAG-S4-rd2 的奖励） |
| rm1__core_sevs1 | critic | 选项 SFT | `outputs/rt/rm/rm1__core_sevs1/` 6070 行 | experiments.md 4.9「sev 筛过的真错 vs critic 拿不准的真错」 |
| rm1__core_sevs2 | critic | 选项 SFT | `outputs/rt/rm/rm1__core_sevs2/` 6070 行 | 同上 |
| rm1__core_sevrnd | critic | 选项 SFT | `outputs/rt/rm/rm1__core_sevrnd/` 6070 行 | 表 3（随机 733） |
| rm1__core_sevcl | critic | 选项 SFT | `outputs/rt/rm/rm1__core_sevcl/` 6226 行 | 表 3（Claude 确认真错 811） |
| rm1__mcgold_ref | critic | 选项 SFT | `outputs/rt/rm/rm1__mcgold_ref/` 6330 行 | 表 3（ref 底座，6330 条） |
| rm1__mcgold_h3ctl | critic | 选项 SFT | `outputs/rt/rm/rm1__mcgold_h3ctl/` 7438 行 | 表 3（ref + 554 对）；表 1（奖励粒度 R0–R3 四组共用的 critic） |
| rm1__mcgold_ref_z1500 | critic | 选项 SFT | `outputs/rt/rm/rm1__mcgold_ref_z1500/` 9330 行 | 表 3（ref + 1500 对） |
| rm1__mcgold_ref_z6000 | critic | 选项 SFT | `outputs/rt/rm/rm1__mcgold_ref_z6000/` 18330 行 | 表 3（ref + 6000 对） |
| rm1__mcgold_hum | critic | 选项 SFT | `outputs/rt/rm/rm1__mcgold_hum/` 3030 行 | 表 3（只用人工标注 3030 条） |
| rm1__mcgold_syn | critic | 选项 SFT | `outputs/rt/rm/rm1__mcgold_syn/` 3030 行 | 表 3（只用合成数据 3030 条） |
| rm1__rm1_gold_9b | critic | 选项 SFT | `outputs/rt/rm/rm1__rm1_gold_9b/` 5817 行 | 表 2、表 8、表 9 的全部 RAG 评测 |
| sev__v3c0 | rm | 选项 SFT | `outputs/rt/rm/sev__v3c0/` 2556 行 | 表 5、表 6（生成器迭代） |
| sev__v3d | rm | 选项 SFT | `outputs/rt/rm/sev__v3d/` 3443 行 | 第二轮补盲的候选打分 |
| rm2__v3b | rm | 选项 SFT | `outputs/rt/rm/rm2__v3b/` 2249 行 | 表 5、表 6；补盲选样的自然度门槛 |
| rm2__v3d | rm | 选项 SFT | `outputs/rt/rm/rm2__v3d/` 3136 行 | 第二轮补盲 |
| sat_pair__core | rm | 选项 SFT | `outputs/rt/rm/sat_pair__core/` 4544 行 | 所有 RAG GRPO 的奖励（只罚） |
| refuse__core | rm | 选项 SFT | `outputs/rt/rm/refuse__core/` 1101 行 | 所有 RAG GRPO 的奖励（0.4 × P） |
| rag-sft-core | rag | SFT | `outputs/rt/core/rag_sft_train.jsonl` | 所有 RAG GRPO 的起点（经 RAG-S1） |
| RAG-S1 | rag | GRPO（R3 奖励） | 起点 rag-sft-core；critic `rm1__core`；200 步 | 表 2 |
| RAG-S2 | rag | GRPO（R3 奖励） | 起点 RAG-S1（mc-grpo-r3_core_coreonly/step200）；critic `rm1__core_r1`；100 步 | 表 2；表 8、表 9 的起点 |
| RAG-S3-r2 | rag | GRPO（R3 奖励） | 起点 RAG-S2；critic `rm1__core_r2`；100 步 | 表 2 |
| RAG-S3-r3 | rag | GRPO（R3 奖励） | 起点 RAG-S2；critic `rm1__core_r3`；100 步 | 表 2 |
| RAG-S3-bf | rag | GRPO（R3 奖励） | 起点 RAG-S2；critic `rm1__core_r2_bfcl`；100 步 | 表 8（核心结论） |
| RAG-S3-r1 | rag | GRPO（R3 奖励） | 起点 RAG-S2；critic `rm1__core_r1`；100 步 | 表 8（核心结论的对照） |
| RAG-S4-rd2 | rag | GRPO（R3 奖励） | 起点 RAG-S3-bf（mc-grpo-r3_core_cont4/step100）；critic `rm1__core_rd2`；100 步 | 表 9 |
| RAG-S1x | rag | GRPO（R3 奖励） | 起点 rag-sft-core；critic `rm1__core_r1`；200 步 | 表 2（对照：新 critic r1 从 SFT 重新训 200 步） |
| gen-v2-mc | gen | SFT | `outputs/rt/core/seed_v2_train.jsonl` | 表 5 |
| gen-v2h2-mc | gen | SFT | `outputs/rt/gen_v2_grpo/sft_hard_v2.jsonl` | 表 5 |
| gen-g1 | gen | GRPO | 起点 gen-v2h2-mc；50 步 | 表 6；两轮补盲的候选采样 |
| gen-v2h3-mc | gen | SFT | `outputs/rt/gen_v2_grpo/sft_hard_v3.jsonl` | 表 6；两轮补盲的候选采样 |

## critic（逐句判有没有幻觉）

### rm1__core

第一版 critic：只用核心集（2B 原回答 + Claude 逐句改对 / 标注）

- **发布位置**：`adapters/rm1__core`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 3（核心集底座）；表 2（RAG-S1 的奖励）
- **训练数据**：`outputs/rt/rm/rm1__core/train.jsonl`、`outputs/rt/rm/rm1__core/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：核心集 outputs/rt/core/core.jsonl、pairs.jsonl（ragtruth.core rm1 构建）
- **训练集**：4604 行（A 有依据 3194，B 矛盾 990，C 无依据 420）；最长 3072 token
- **校准温度**：1.2（校准集原始 AUC 0.863）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准

重训：

```bash
RM=rm1__core GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm1__core_r1

循环一：rm1__core + 生成器 v0 造的错句（sev 门槛 0.5、RM2 自然度 ≥ 0.3，不限 critic 区间）

- **发布位置**：`adapters/rm1__core_r1`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 2（RAG-S2、RAG-S1x 的奖励）；表 8（对照组 RAG-S3-r1 的奖励）
- **训练数据**：`outputs/rt/rm/rm1__core_r1/train.jsonl`、`outputs/rt/rm/rm1__core_r1/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：rm1__core 的训练集 + outputs/rt/core/harvest_r1.jsonl 中 550 条错句 + 550 条对应原句
- **训练集**：5704 行（A 有依据 3744，B 矛盾 1317，C 无依据 643）；最长 3072 token
- **构建参数（meta.json）**：base=rm1__core，added_errors=550，added_originals=550，harvest=outputs/rt/core/harvest_r1.jsonl，band=[0.0, 1.01]
- **校准温度**：1.2（校准集原始 AUC 0.856）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准

重训：

```bash
RM=rm1__core_r1 GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm1__core_r2

只加 critic 0.1–0.7 区间（边界）的错句

- **发布位置**：`adapters/rm1__core_r2`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 3 最后一行（critic 拿不准的真错 581）；表 4、表 7 的底座；表 2（RAG-S3-r2 的奖励）
- **训练数据**：`outputs/rt/rm/rm1__core_r2/train.jsonl`、`outputs/rt/rm/rm1__core_r2/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：rm1__core + harvest_r1 中经 Claude 核实为真错、且 critic 拿不准（盲区 P < 0.3 加边界 0.3–0.7）的 581 条错句 + 原句（id 列表 outputs/rt/core/r2_ids.tsv）
- **训练集**：5766 行（A 有依据 3775，B 矛盾 1423，C 无依据 568）；最长 3072 token
- **构建参数（meta.json）**：base=rm1__core，added_errors=581，added_originals=581，harvest=outputs/rt/core/harvest_r1.jsonl，band=[0.1, 0.7]
- **校准温度**：1.35（校准集原始 AUC 0.856）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准

重训：

```bash
RM=rm1__core_r2 GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm1__core_r3

长句 / 列表句增强（r2 + 2000 条合并句）

- **发布位置**：`adapters/rm1__core_r3`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 2（RAG-S3-r3 的奖励）
- **训练数据**：`outputs/rt/rm/rm1__core_r3/train.jsonl`、`outputs/rt/rm/rm1__core_r3/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：rm1__core_r2 + 2000 条长句 / 列表句（留出 310）
- **训练集**：7766 行（A 有依据 4775，B 矛盾 2125，C 无依据 866）；最长 3072 token
- **构建参数（meta.json）**：base=rm1__core_r2，added_errors=581，added_originals=581，harvest=outputs/rt/core/harvest_r1.jsonl，band=[0.1, 0.7]，merge_added=2000
- **校准温度**：1.4（校准集原始 AUC 0.863）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准

重训：

```bash
RM=rm1__core_r3 GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm1__core_r2_sccl

单类型补盲（范围偏移）：Claude 核实过的 285 条 + 原句

- **发布位置**：`adapters/rm1__core_r2_sccl`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 4（单类型补盲，核实组）
- **训练数据**：`outputs/rt/rm/rm1__core_r2_sccl/train.jsonl`、`outputs/rt/rm/rm1__core_r2_sccl/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：rm1__core_r2 + outputs/rt/core/scope_label 中 Claude 判真错且自然的 285 条
- **训练集**：6336 行（A 有依据 4060，B 矛盾 1704，C 无依据 572）；最长 3072 token
- **构建参数（meta.json）**：base=rm1__core_r2，added_errors=285，added_originals=581，harvest=outputs/rt/core/harvest_r1.jsonl，band=[0.1, 0.7]，arm=cl，labeled=595
- **校准温度**：1.3（校准集原始 AUC 0.870）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准

重训：

```bash
RM=rm1__core_r2_sccl GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm1__core_r2_scnc

单类型补盲对照：同数量、不经 Claude 核实

- **发布位置**：`adapters/rm1__core_r2_scnc`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 4（不核实组）
- **训练数据**：`outputs/rt/rm/rm1__core_r2_scnc/train.jsonl`、`outputs/rt/rm/rm1__core_r2_scnc/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：rm1__core_r2 + 同候选池随机 285 条（不核实，真合格约 50%）
- **训练集**：6336 行（A 有依据 4060，B 矛盾 1708，C 无依据 568）；最长 3072 token
- **构建参数（meta.json）**：base=rm1__core_r2，added_errors=285，added_originals=581，harvest=outputs/rt/core/harvest_r1.jsonl，band=[0.1, 0.7]，arm=nc，labeled=595
- **校准温度**：1.2（校准集原始 AUC 0.862）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准

重训：

```bash
RM=rm1__core_r2_scnc GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm1__core_r2_sccl_no

单类型补盲：只加错句、不配原句

- **发布位置**：`adapters/rm1__core_r2_sccl_no`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 4（配不配原句）
- **训练数据**：`outputs/rt/rm/rm1__core_r2_sccl_no/train.jsonl`、`outputs/rt/rm/rm1__core_r2_sccl_no/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：同 sccl，但不加对应原句
- **训练集**：6051 行（A 有依据 3775，B 矛盾 1704，C 无依据 572）；最长 3072 token
- **构建参数（meta.json）**：base=rm1__core_r2，added_errors=285，added_originals=581，harvest=outputs/rt/core/harvest_r1.jsonl，band=[0.1, 0.7]，arm=cl，labeled=595
- **校准温度**：1.3（校准集原始 AUC 0.856）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准

重训：

```bash
RM=rm1__core_r2_sccl_no GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm1__core_r2_bfcl

第一轮闭环 critic：多类型补盲（范围偏移 / 拼接 / 近义 / 张冠李戴），Claude 核实

- **发布位置**：`adapters/rm1__core_r2_bfcl`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 7、表 8（RAG-S3-bf 的奖励）
- **训练数据**：`outputs/rt/rm/rm1__core_r2_bfcl/train.jsonl`、`outputs/rt/rm/rm1__core_r2_bfcl/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：rm1__core_r2 + outputs/rt/gen_v2_grpo/blind_fill（787 条候选经 Claude 标注；保留真错且自然的 558 条：主池 429 + 对照池 129）；只加错句
- **训练集**：6324 行（A 有依据 3775，B 矛盾 1916，C 无依据 633）；最长 3072 token
- **构建参数（meta.json）**：base=rm1__core_r2，added_errors=558，added_originals=581，harvest=outputs/rt/core/harvest_r1.jsonl，band=[0.1, 0.7]，arm=cl，labeled=787，verify_kept=429
- **校准温度**：1.35（校准集原始 AUC 0.850）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准

重训：

```bash
RM=rm1__core_r2_bfcl GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm1__core_r2_bfnc

第一轮闭环对照：同数量、不经 Claude 核实

- **发布位置**：`adapters/rm1__core_r2_bfnc`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 7（核实 vs 不核实）
- **训练数据**：`outputs/rt/rm/rm1__core_r2_bfnc/train.jsonl`、`outputs/rt/rm/rm1__core_r2_bfnc/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：rm1__core_r2 + blind_fill 同候选池同数量，不核实
- **训练集**：6324 行（A 有依据 3775，B 矛盾 1922，C 无依据 627）；最长 3072 token
- **构建参数（meta.json）**：base=rm1__core_r2，added_errors=558，added_originals=581，harvest=outputs/rt/core/harvest_r1.jsonl，band=[0.1, 0.7]，arm=nc，labeled=787，verify_kept=429
- **校准温度**：1.35（校准集原始 AUC 0.844）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准

重训：

```bash
RM=rm1__core_r2_bfnc GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm1__core_rd2

第二轮闭环 critic：在 bfcl 基础上加第二轮补盲

- **发布位置**：`adapters/rm1__core_rd2`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 9（静态评测；RAG-S4-rd2 的奖励）
- **训练数据**：`outputs/rt/rm/rm1__core_rd2/train.jsonl`、`outputs/rt/rm/rm1__core_rd2/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：rm1__core_r2_bfcl + outputs/rt/gen_v2_grpo/blind_fill_r2（1481 条 Claude 标注，真错 + 自然 799 条）
- **训练集**：7123 行（A 有依据 3775，B 矛盾 2610，C 无依据 738）；最长 3072 token
- **构建参数（meta.json）**：base=rm1__core_r2_bfcl，added_errors=799，added_originals=581，harvest=outputs/rt/core/harvest_r1.jsonl，band=[0.1, 0.7]，arm=cl，labeled=1481，verify_kept=799
- **校准温度**：1.4（校准集原始 AUC 0.833）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准

重训：

```bash
RM=rm1__core_rd2 GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm1__core_sevs1

sev 筛选对照：旧 sev 按分数取 733 条

- **发布位置**：`adapters/rm1__core_sevs1`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：experiments.md 4.9「sev 筛过的真错 vs critic 拿不准的真错」
- **训练数据**：`outputs/rt/rm/rm1__core_sevs1/train.jsonl`、`outputs/rt/rm/rm1__core_sevs1/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：rm1__core + 候选池 1370 条中 sev1 打分最高的 733 条
- **训练集**：6070 行（A 有依据 3927，B 矛盾 1346，C 无依据 797）；最长 3072 token
- **构建参数（meta.json）**：base=rm1__core，added_errors=733
- **校准温度**：1.25（校准集原始 AUC 0.858）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准

重训：

```bash
RM=rm1__core_sevs1 GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm1__core_sevs2

sev 筛选对照：新 sev 门槛 0.8

- **发布位置**：`adapters/rm1__core_sevs2`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：同上
- **训练数据**：`outputs/rt/rm/rm1__core_sevs2/train.jsonl`、`outputs/rt/rm/rm1__core_sevs2/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：rm1__core + 候选池中 sev2 ≥ 0.8 的 770 条里取 733
- **训练集**：6070 行（A 有依据 3927，B 矛盾 1319，C 无依据 824）；最长 3072 token
- **构建参数（meta.json）**：base=rm1__core，added_errors=733
- **校准温度**：1.35（校准集原始 AUC 0.849）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准

重训：

```bash
RM=rm1__core_sevs2 GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm1__core_sevrnd

sev 筛选对照：不过滤，随机 733 条

- **发布位置**：`adapters/rm1__core_sevrnd`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 3（随机 733）
- **训练数据**：`outputs/rt/rm/rm1__core_sevrnd/train.jsonl`、`outputs/rt/rm/rm1__core_sevrnd/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：rm1__core + 候选池随机 733 条
- **训练集**：6070 行（A 有依据 3927，B 矛盾 1346，C 无依据 797）；最长 3072 token
- **构建参数（meta.json）**：base=rm1__core，added_errors=733
- **校准温度**：1.2（校准集原始 AUC 0.852）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准

重训：

```bash
RM=rm1__core_sevrnd GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm1__core_sevcl

sev 筛选对照：Claude 判真错的 811 条

- **发布位置**：`adapters/rm1__core_sevcl`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 3（Claude 确认真错 811）
- **训练数据**：`outputs/rt/rm/rm1__core_sevcl/train.jsonl`、`outputs/rt/rm/rm1__core_sevcl/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：rm1__core + 候选池 2622 条中 Claude 判真错的 811 条
- **训练集**：6226 行（A 有依据 4005，B 矛盾 1479，C 无依据 742）；最长 3072 token
- **构建参数（meta.json）**：base=rm1__core，added_errors=811
- **校准温度**：1.4（校准集原始 AUC 0.855）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准

重训：

```bash
RM=rm1__core_sevcl GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm1__mcgold_ref

早期 critic（RAGTruth 人工标注 + 9B 回答 Claude 金标 + 拒答句）

- **发布位置**：`adapters/rm1__mcgold_ref`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 3（ref 底座，6330 条）
- **训练数据**：`outputs/rt/rm/rm1__mcgold_ref/train.jsonl`、`outputs/rt/rm/rm1__mcgold_ref/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：rm1__rm1_gold_9b 的训练集 + 10 句拒答改标 + refuse 训练切分原文各一条拒答（513 句；ragtruth/critic_refuse.py）
- **训练集**：6330 行（A 有依据 3828，B 矛盾 1212，C 无依据 1290）；最长 3072 token
- **构建参数（meta.json）**：base=rm1__mcgold
- **校准温度**：1.15（校准集原始 AUC 0.969）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准

重训：

```bash
RM=rm1__mcgold_ref GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm1__mcgold_h3ctl

rm1__mcgold_ref + 生成器 h3 造的 554 条错句（不限 critic 区间）

- **发布位置**：`adapters/rm1__mcgold_h3ctl`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 3（ref + 554 对）；表 1（奖励粒度 R0–R3 四组共用的 critic）
- **训练数据**：`outputs/rt/rm/rm1__mcgold_h3ctl/train.jsonl`、`outputs/rt/rm/rm1__mcgold_h3ctl/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：rm1__mcgold_ref + outputs/rt/gentarget/harvest_h3.jsonl 554 条错句 + 原句
- **训练集**：7438 行（A 有依据 4382，B 矛盾 1503，C 无依据 1553）；最长 3072 token
- **构建参数（meta.json）**：base=rm1__mcgold_ref，added_errors=554，added_originals=554，harvest=outputs/rt/gentarget/harvest_h3.jsonl，band=[0.0, 1.01]
- **校准温度**：1.05（校准集原始 AUC 0.972）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准

重训：

```bash
RM=rm1__mcgold_h3ctl GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm1__mcgold_ref_z1500

合成数据量对照：+1500 条（不加自然度门槛）

- **发布位置**：`adapters/rm1__mcgold_ref_z1500`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 3（ref + 1500 对）
- **训练数据**：`outputs/rt/rm/rm1__mcgold_ref_z1500/train.jsonl`、`outputs/rt/rm/rm1__mcgold_ref_z1500/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：rm1__mcgold_ref + harvest_h3 1500 条
- **训练集**：9330 行（A 有依据 5328，B 矛盾 2007，C 无依据 1995）；最长 3072 token
- **构建参数（meta.json）**：base=rm1__mcgold_ref，added_errors=1500，added_originals=1500，harvest=outputs/rt/gentarget/harvest_h3.jsonl，band=[0.0, 1.01]
- **校准温度**：1.2（校准集原始 AUC 0.975）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准

重训：

```bash
RM=rm1__mcgold_ref_z1500 GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm1__mcgold_ref_z6000

合成数据量对照：+6000 条

- **发布位置**：`adapters/rm1__mcgold_ref_z6000`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 3（ref + 6000 对）
- **训练数据**：`outputs/rt/rm/rm1__mcgold_ref_z6000/train.jsonl`、`outputs/rt/rm/rm1__mcgold_ref_z6000/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：rm1__mcgold_ref + harvest_h3 6000 条
- **训练集**：18330 行（A 有依据 9828，B 矛盾 4296，C 无依据 4206）；最长 3072 token
- **构建参数（meta.json）**：base=rm1__mcgold_ref，added_errors=6000，added_originals=6000，harvest=outputs/rt/gentarget/harvest_h3.jsonl，band=[0.0, 1.01]
- **校准温度**：1.15（校准集原始 AUC 0.969）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准

重训：

```bash
RM=rm1__mcgold_ref_z6000 GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm1__mcgold_hum

纯人工数据（3030 行）

- **发布位置**：`adapters/rm1__mcgold_hum`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 3（只用人工标注 3030 条）
- **训练数据**：`outputs/rt/rm/rm1__mcgold_hum/train.jsonl`、`outputs/rt/rm/rm1__mcgold_hum/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：rm1__mcgold_ref 中人工标注部分 2517 + 拒答 513
- **训练集**：3030 行（A 有依据 1869，B 矛盾 701，C 无依据 460）；最长 3072 token
- **构建参数（meta.json）**：base=rm1__mcgold_ref
- **校准温度**：1.35（校准集原始 AUC 0.969）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准

重训：

```bash
RM=rm1__mcgold_hum GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm1__mcgold_syn

纯合成数据（同 3030 行）

- **发布位置**：`adapters/rm1__mcgold_syn`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 3（只用合成数据 3030 条）
- **训练数据**：`outputs/rt/rm/rm1__mcgold_syn/train.jsonl`、`outputs/rt/rm/rm1__mcgold_syn/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：harvest_h3 1500 条错句 + 原句 + 拒答
- **训练集**：3030 行（A 有依据 1328，B 矛盾 995，C 无依据 707）；最长 3072 token
- **构建参数（meta.json）**：base=rm1__mcgold_ref_n1500，added_errors=1500，added_originals=1500，harvest=outputs/rt/gentarget/harvest_h3.jsonl，band=[0.0, 1.01]
- **校准温度**：1.5（校准集原始 AUC 0.954）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准

重训：

```bash
RM=rm1__mcgold_syn GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm1__rm1_gold_9b

9B 金标 critic：只用于评测（RAG 各版本的含显然错 / 任意错比例），不进任何训练奖励

- **发布位置**：`adapters/rm1__rm1_gold_9b`
- **基座**：Qwen/Qwen3.5-9B
- **对应结论**：表 2、表 8、表 9 的全部 RAG 评测
- **训练数据**：`outputs/rt/rm/rm1__rm1_gold_9b/train.jsonl`、`outputs/rt/rm/rm1__rm1_gold_9b/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：RAGTruth 人工标注 + 9B 在训练池 300 个原文上的回答（Claude 逐句金标 1479 行）
- **训练集**：5817 行（A 有依据 3724，B 矛盾 803，C 无依据 1290）；最长 3072 token
- **校准温度**：1.3
- **训练配置**：`configs/rt/rm_9b.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，批 1 × 累积 16，bf16，seed 0（LoRA 同 2B 版）
- **输出**：单个字母：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准

重训：

```bash
RM=rm1__rm1_gold_9b bash scripts/remote/run_rmtrain.sh（9B 版，超参 configs/rt/rm_9b.yaml）
```


## 其他打分模型

### sev__v3c0

sev：给"原句 → 改写句"判错误严重程度 / 要求的错误类型（9 选 1）；生成器 GRPO 的门槛和奖励

- **发布位置**：`adapters/sev__v3c0`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 5、表 6（生成器迭代）
- **训练数据**：`outputs/rt/rm/sev__v3c0/train.jsonl`、`outputs/rt/rm/sev__v3c0/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：生成器 v2 输出的 Claude 标注（outputs/rt/core/gen_v2_label*）
- **训练集**：2556 行（A 225，B 373，C 1091，D 867）；最长 4096 token
- **校准温度**：1.65（校准集原始 AUC 0.818）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 要求的那种·显然 / B 要求的那种·细微 / C 别的错 / D 没错；P(有错) = A + B + C，按温度校准

重训：

```bash
RM=sev__v3c0 GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### sev__v3d

sev v3c0 的配方 + 第一轮盲区的 Claude 标注（sevaug_r1，887 行）

- **发布位置**：`adapters/sev__v3d`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：第二轮补盲的候选打分
- **训练数据**：`outputs/rt/rm/sev__v3d/train.jsonl`、`outputs/rt/rm/sev__v3d/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：sev__v3c0 训练集 + outputs/rt/core/sevaug_r1
- **训练集**：3443 行（A 367，B 709，C 1277，D 1090）；最长 4096 token
- **校准温度**：1.55（校准集原始 AUC 0.814）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 要求的那种·显然 / B 要求的那种·细微 / C 别的错 / D 没错；P(有错) = A + B + C，按温度校准

重训：

```bash
RM=sev__v3d GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm2__v3b

RM2 自然度：改写句读起来是否像模型自然写出的

- **发布位置**：`adapters/rm2__v3b`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 5、表 6；补盲选样的自然度门槛
- **训练数据**：`outputs/rt/rm/rm2__v3b/train.jsonl`、`outputs/rt/rm/rm2__v3b/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：生成器样本的 Claude 自然度标注
- **训练集**：2249 行（A 1830，B 419）；最长 3072 token
- **校准温度**：1.4（校准集原始 AUC 0.847）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 自然 / B 不自然；P(自然) = P(A)，按温度校准

重训：

```bash
RM=rm2__v3b GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### rm2__v3d

RM2 v3b 的配方 + sevaug_r1

- **发布位置**：`adapters/rm2__v3d`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：第二轮补盲
- **训练数据**：`outputs/rt/rm/rm2__v3d/train.jsonl`、`outputs/rt/rm/rm2__v3d/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：rm2__v3b 训练集 + sevaug_r1
- **训练集**：3136 行（A 2647，B 489）；最长 3072 token
- **校准温度**：1.15（校准集原始 AUC 0.886）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 自然 / B 不自然；P(自然) = P(A)，按温度校准

重训：

```bash
RM=rm2__v3d GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### sat_pair__core

满意度（相对）：两条回答哪条更好，只用来罚，防钻空子

- **发布位置**：`adapters/sat_pair__core`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：所有 RAG GRPO 的奖励（只罚）
- **训练数据**：`outputs/rt/rm/sat_pair__core/train.jsonl`、`outputs/rt/rm/sat_pair__core/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：核心集 Claude 满意度标注（label_raw.jsonl）
- **训练集**：4544 行（A 2272，B 2272）；最长 4096 token
- **校准温度**：0.2（校准集原始 AUC 1.000）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 参考回答更好 / B 新回答更好；P_sat = P(B)，按温度校准

重训：

```bash
RM=sat_pair__core GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```

### refuse__core

拒答：原文是否确实回答不了

- **发布位置**：`adapters/refuse__core`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：所有 RAG GRPO 的奖励（0.4 × P）
- **训练数据**：`outputs/rt/rm/refuse__core/train.jsonl`、`outputs/rt/rm/refuse__core/calib.jsonl（只用来拟合温度）`
- **数据怎么来的**：核心集拒答标注
- **训练集**：1101 行（A 668，B 433）；最长 3072 token
- **构建参数（meta.json）**：threshold=0.3
- **校准温度**：2.35（校准集原始 AUC 0.947）
- **训练配置**：`configs/rt/rm_mc.yaml`；LoRA r 16 / alpha 32 / all-linear；2 轮，lr 1e-4 cosine，warmup 3%，批 4 × 累积 4 = 16，bf16，seed 0
- **输出**：单个字母：A 原文能回答 / B 原文答不了；P(可以拒答) = P(B)，按温度校准

重训：

```bash
RM=refuse__core GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh
```


## RAG 模型（被训的回答模型）

### rag-sft-core

RAG 监督微调：核心集里 Claude 改对后的回答（726 条）

- **发布位置**：`adapters/rag-sft-core`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：所有 RAG GRPO 的起点（经 RAG-S1）
- **训练数据**：`outputs/rt/core/rag_sft_train.jsonl`（726 行）
- **训练配置**：`configs/rt/sft_policy_mc.yaml`；LoRA r 32 / alpha 64（q k v o gate up down），2 轮，lr 5e-5 cosine，批 4 × 累积 4，max_length 4096，seed 0

重训：

```bash
python -m train.sft --config <sft_policy_mc.yaml，data 换成 outputs/rt/core/rag_sft_train.jsonl> --mode coldstart（scripts/remote/run_core1.sh 的 sft 函数）
```

### RAG-S1

第 1 段强化

- **发布位置**：`adapters/mc-grpo-r3_core_coreonly/step200`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 2
- **起点**：rag-sft-core
- **奖励里的 critic**：`rm1__core`
- **训练**：200 步；lr 5e-6，每步 16 个原文 × 每个 16 条回答，KL 相对起点，每 50 步存检查点，TripWire 开
- **奖励**：R3：critic 逐句 P(幻觉) 去底噪计数 K = max(2, 0.5 × 参考句数) + 句子层 β 0.6 + 前一句 γ 0.3；满意度 sat_pair__core 只罚（0.8 ×，最多 0.4）；拒答 refuse__core 0.4 × P；硬规则（长度、语言、照抄）
- **训练原文**：outputs/rt/core/grpo_train.jsonl（750 个原文，去掉核心集留出原文）+ outputs/rt/minicpm/grpo/dev.jsonl；起点重采参考回答（ans_data ref / ref2）→ outputs/rt/core/loop2_prompts_<起点>/
- **评测**：起点 + 第 50 / 100（/ 200）步在三套验收原文上生成（grpo300 T1.0、grpo300b / grpo300c T0.7，top_p 0.95），9B 金标 critic 逐句打分 → outputs/rt/core/loop2_<组>/eval/
- **训练日志**：`outputs/rt/core/loop2_<组>/run/log.jsonl`（逐步）、`quick_eval.jsonl`

重训：

```bash
ARM=coreonly CRITIC=rm1__core bash scripts/remote/run_loop2.sh
```

### RAG-S2

第 2 段强化（换循环一 critic）

- **发布位置**：`adapters/mc-grpo-r3_core_cont/step100`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 2；表 8、表 9 的起点
- **起点**：RAG-S1（mc-grpo-r3_core_coreonly/step200）
- **奖励里的 critic**：`rm1__core_r1`
- **训练**：100 步；lr 5e-6，每步 16 个原文 × 每个 16 条回答，KL 相对起点，每 50 步存检查点，TripWire 开
- **奖励**：R3：critic 逐句 P(幻觉) 去底噪计数 K = max(2, 0.5 × 参考句数) + 句子层 β 0.6 + 前一句 γ 0.3；满意度 sat_pair__core 只罚（0.8 ×，最多 0.4）；拒答 refuse__core 0.4 × P；硬规则（长度、语言、照抄）
- **训练原文**：outputs/rt/core/grpo_train.jsonl（750 个原文，去掉核心集留出原文）+ outputs/rt/minicpm/grpo/dev.jsonl；起点重采参考回答（ans_data ref / ref2）→ outputs/rt/core/loop2_prompts_<起点>/
- **评测**：起点 + 第 50 / 100（/ 200）步在三套验收原文上生成（grpo300 T1.0、grpo300b / grpo300c T0.7，top_p 0.95），9B 金标 critic 逐句打分 → outputs/rt/core/loop2_<组>/eval/
- **训练日志**：`outputs/rt/core/loop2_<组>/run/log.jsonl`（逐步）、`quick_eval.jsonl`

重训：

```bash
START=adapters/mc-grpo-r3_core_coreonly/step200 CRITIC=rm1__core_r1 ARM=cont bash scripts/remote/run_loop2cont.sh
```

### RAG-S3-r2

第 3 段平行尝试：r2 当奖励

- **发布位置**：`adapters/mc-grpo-r3_core_cont2/step100`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 2
- **起点**：RAG-S2
- **奖励里的 critic**：`rm1__core_r2`
- **训练**：100 步；lr 5e-6，每步 16 个原文 × 每个 16 条回答，KL 相对起点，每 50 步存检查点，TripWire 开
- **奖励**：R3：critic 逐句 P(幻觉) 去底噪计数 K = max(2, 0.5 × 参考句数) + 句子层 β 0.6 + 前一句 γ 0.3；满意度 sat_pair__core 只罚（0.8 ×，最多 0.4）；拒答 refuse__core 0.4 × P；硬规则（长度、语言、照抄）
- **训练原文**：outputs/rt/core/grpo_train.jsonl（750 个原文，去掉核心集留出原文）+ outputs/rt/minicpm/grpo/dev.jsonl；起点重采参考回答（ans_data ref / ref2）→ outputs/rt/core/loop2_prompts_<起点>/
- **评测**：起点 + 第 50 / 100（/ 200）步在三套验收原文上生成（grpo300 T1.0、grpo300b / grpo300c T0.7，top_p 0.95），9B 金标 critic 逐句打分 → outputs/rt/core/loop2_<组>/eval/
- **训练日志**：`outputs/rt/core/loop2_<组>/run/log.jsonl`（逐步）、`quick_eval.jsonl`

重训：

```bash
START=adapters/mc-grpo-r3_core_cont/step100 CRITIC=rm1__core_r2 ARM=cont2 bash scripts/remote/run_loop2cont.sh
```

### RAG-S3-r3

第 3 段平行尝试：r3 当奖励

- **发布位置**：`adapters/mc-grpo-r3_core_cont3/step100`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 2
- **起点**：RAG-S2
- **奖励里的 critic**：`rm1__core_r3`
- **训练**：100 步；lr 5e-6，每步 16 个原文 × 每个 16 条回答，KL 相对起点，每 50 步存检查点，TripWire 开
- **奖励**：R3：critic 逐句 P(幻觉) 去底噪计数 K = max(2, 0.5 × 参考句数) + 句子层 β 0.6 + 前一句 γ 0.3；满意度 sat_pair__core 只罚（0.8 ×，最多 0.4）；拒答 refuse__core 0.4 × P；硬规则（长度、语言、照抄）
- **训练原文**：outputs/rt/core/grpo_train.jsonl（750 个原文，去掉核心集留出原文）+ outputs/rt/minicpm/grpo/dev.jsonl；起点重采参考回答（ans_data ref / ref2）→ outputs/rt/core/loop2_prompts_<起点>/
- **评测**：起点 + 第 50 / 100（/ 200）步在三套验收原文上生成（grpo300 T1.0、grpo300b / grpo300c T0.7，top_p 0.95），9B 金标 critic 逐句打分 → outputs/rt/core/loop2_<组>/eval/
- **训练日志**：`outputs/rt/core/loop2_<组>/run/log.jsonl`（逐步）、`quick_eval.jsonl`

重训：

```bash
START=adapters/mc-grpo-r3_core_cont/step100 CRITIC=rm1__core_r3 ARM=cont3 bash scripts/remote/run_loop2cont.sh
```

### RAG-S3-bf

第一轮闭环：补盲后的 critic 当奖励

- **发布位置**：`adapters/mc-grpo-r3_core_cont4/step100`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 8（核心结论）
- **起点**：RAG-S2
- **奖励里的 critic**：`rm1__core_r2_bfcl`
- **训练**：100 步；lr 5e-6，每步 16 个原文 × 每个 16 条回答，KL 相对起点，每 50 步存检查点，TripWire 开
- **奖励**：R3：critic 逐句 P(幻觉) 去底噪计数 K = max(2, 0.5 × 参考句数) + 句子层 β 0.6 + 前一句 γ 0.3；满意度 sat_pair__core 只罚（0.8 ×，最多 0.4）；拒答 refuse__core 0.4 × P；硬规则（长度、语言、照抄）
- **训练原文**：outputs/rt/core/grpo_train.jsonl（750 个原文，去掉核心集留出原文）+ outputs/rt/minicpm/grpo/dev.jsonl；起点重采参考回答（ans_data ref / ref2）→ outputs/rt/core/loop2_prompts_<起点>/
- **评测**：起点 + 第 50 / 100（/ 200）步在三套验收原文上生成（grpo300 T1.0、grpo300b / grpo300c T0.7，top_p 0.95），9B 金标 critic 逐句打分 → outputs/rt/core/loop2_<组>/eval/
- **训练日志**：`outputs/rt/core/loop2_<组>/run/log.jsonl`（逐步）、`quick_eval.jsonl`

重训：

```bash
START=adapters/mc-grpo-r3_core_cont/step100 CRITIC=rm1__core_r2_bfcl ARM=cont4 bash scripts/remote/run_loop2cont.sh
```

### RAG-S3-r1

对照：奖励模型不换，同样多训 100 步

- **发布位置**：`adapters/mc-grpo-r3_core_cont5/step100`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 8（核心结论的对照）
- **起点**：RAG-S2
- **奖励里的 critic**：`rm1__core_r1`
- **训练**：100 步；lr 5e-6，每步 16 个原文 × 每个 16 条回答，KL 相对起点，每 50 步存检查点，TripWire 开
- **奖励**：R3：critic 逐句 P(幻觉) 去底噪计数 K = max(2, 0.5 × 参考句数) + 句子层 β 0.6 + 前一句 γ 0.3；满意度 sat_pair__core 只罚（0.8 ×，最多 0.4）；拒答 refuse__core 0.4 × P；硬规则（长度、语言、照抄）
- **训练原文**：outputs/rt/core/grpo_train.jsonl（750 个原文，去掉核心集留出原文）+ outputs/rt/minicpm/grpo/dev.jsonl；起点重采参考回答（ans_data ref / ref2）→ outputs/rt/core/loop2_prompts_<起点>/
- **评测**：起点 + 第 50 / 100（/ 200）步在三套验收原文上生成（grpo300 T1.0、grpo300b / grpo300c T0.7，top_p 0.95），9B 金标 critic 逐句打分 → outputs/rt/core/loop2_<组>/eval/
- **训练日志**：`outputs/rt/core/loop2_<组>/run/log.jsonl`（逐步）、`quick_eval.jsonl`

重训：

```bash
START=adapters/mc-grpo-r3_core_cont/step100 CRITIC=rm1__core_r1 ARM=cont5 bash scripts/remote/run_loop2cont.sh
```

### RAG-S4-rd2

第二轮闭环

- **发布位置**：`adapters/mc-grpo-r3_core_cont6/step100`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 9
- **起点**：RAG-S3-bf（mc-grpo-r3_core_cont4/step100）
- **奖励里的 critic**：`rm1__core_rd2`
- **训练**：100 步；lr 5e-6，每步 16 个原文 × 每个 16 条回答，KL 相对起点，每 50 步存检查点，TripWire 开
- **奖励**：R3：critic 逐句 P(幻觉) 去底噪计数 K = max(2, 0.5 × 参考句数) + 句子层 β 0.6 + 前一句 γ 0.3；满意度 sat_pair__core 只罚（0.8 ×，最多 0.4）；拒答 refuse__core 0.4 × P；硬规则（长度、语言、照抄）
- **训练原文**：outputs/rt/core/grpo_train.jsonl（750 个原文，去掉核心集留出原文）+ outputs/rt/minicpm/grpo/dev.jsonl；起点重采参考回答（ans_data ref / ref2）→ outputs/rt/core/loop2_prompts_<起点>/
- **评测**：起点 + 第 50 / 100（/ 200）步在三套验收原文上生成（grpo300 T1.0、grpo300b / grpo300c T0.7，top_p 0.95），9B 金标 critic 逐句打分 → outputs/rt/core/loop2_<组>/eval/
- **训练日志**：`outputs/rt/core/loop2_<组>/run/log.jsonl`（逐步）、`quick_eval.jsonl`

重训：

```bash
START=adapters/mc-grpo-r3_core_cont4/step100 CRITIC=rm1__core_rd2 ARM=cont6 bash scripts/remote/run_loop2cont.sh
```

### RAG-S1x

对照：用同一个新 critic r1，从 SFT 重新训 200 步（不分段）

- **发布位置**：`adapters/mc-grpo-r3_core_main/step200`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 2（对照：新 critic r1 从 SFT 重新训 200 步）
- **起点**：rag-sft-core
- **奖励里的 critic**：`rm1__core_r1`
- **训练**：200 步；lr 5e-6，每步 16 个原文 × 每个 16 条回答，KL 相对起点，每 50 步存检查点，TripWire 开
- **奖励**：R3：critic 逐句 P(幻觉) 去底噪计数 K = max(2, 0.5 × 参考句数) + 句子层 β 0.6 + 前一句 γ 0.3；满意度 sat_pair__core 只罚（0.8 ×，最多 0.4）；拒答 refuse__core 0.4 × P；硬规则（长度、语言、照抄）
- **训练原文**：outputs/rt/core/grpo_train.jsonl（750 个原文，去掉核心集留出原文）+ outputs/rt/minicpm/grpo/dev.jsonl；起点重采参考回答（ans_data ref / ref2）→ outputs/rt/core/loop2_prompts_<起点>/
- **评测**：起点 + 第 50 / 100（/ 200）步在三套验收原文上生成（grpo300 T1.0、grpo300b / grpo300c T0.7，top_p 0.95），9B 金标 critic 逐句打分 → outputs/rt/core/loop2_<组>/eval/
- **训练日志**：`outputs/rt/core/loop2_<组>/run/log.jsonl`（逐步）、`quick_eval.jsonl`

重训：

```bash
ARM=main CRITIC=rm1__core_r1 bash scripts/remote/run_loop2.sh
```


## 错误生成器

### gen-v2-mc

错误生成器 v2：输入原文 + 完整回答 + 9 种错误机制之一，输出第几句 + 改动 + 改写句

- **发布位置**：`adapters/gen-v2`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 5
- **训练数据**：`outputs/rt/core/seed_v2_train.jsonl`（1454 行）
- **训练配置**：`configs/rt/sft_gen_v2_mc.yaml`；LoRA r 16 / alpha 32，1 轮，lr 5e-5，批 2 × 累积 2，max_length 4096

重训：

```bash
python -m train.sft --config configs/rt/sft_gen_v2_mc.yaml --mode coldstart
```

### gen-v2h2-mc

难样本 SFT 第二轮（g1 的起点）

- **发布位置**：`adapters/gen-v2h2`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 5
- **训练数据**：`outputs/rt/gen_v2_grpo/sft_hard_v2.jsonl`（1962 行）
- **训练配置**：`configs/rt/sft_gen_v2h2_mc.yaml`；同 gen-v2-mc

重训：

```bash
python -m train.sft --config configs/rt/sft_gen_v2h2_mc.yaml --mode coldstart
```

### gen-g1

生成器 GRPO：专门造 critic 拿不准（P 0.3–0.5）的错句

- **发布位置**：`adapters/gen-g1/step50`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 6；两轮补盲的候选采样
- **起点**：gen-v2h2-mc
- **训练**：50 步
- **奖励**：过门槛（sev 有错 ≥ 0.9、RM2 自然 ≥ 0.5）后按 critic rm1__core_r2 的 P 给三角奖励：0.3–0.5 为 1，0.5 → 0.9 线性降到 0；没过门槛 0

重训：

```bash
python -m train.gen_grpo --causal 1 --base <MiniCPM5-2B> --init adapters/gen-v2h2 --prompts outputs/rt/gen_v2_grpo/prompts_w.jsonl --reward v2sev --temps sev=sev__v3c0,rm2v3=rm2__v3b,rm1=rm1__core_r2 --max_tokens 512 --steps 50 --lr 5e-6 --n 32 --prompts_per_step 8 --v2_shape hard --err_gate 0.9 --nat_gate 0.5 --hard_shape tri --hard_base 0（scripts/remote/run_gen_c.sh）
```

### gen-v2h3-mc

难样本 SFT 第三轮：原 SFT 数据 + g1 采样里过门槛、critic 0.1–0.5、类型对的样本

- **发布位置**：`adapters/gen-v2h3`
- **基座**：openbmb/MiniCPM5-2B@f974000
- **对应结论**：表 6；两轮补盲的候选采样
- **训练数据**：`outputs/rt/gen_v2_grpo/sft_hard_v3.jsonl`（2280 行）
- **训练配置**：`configs/rt/sft_gen_v2h3_mc.yaml`；同 gen-v2-mc

重训：

```bash
python -m train.sft --config configs/rt/sft_gen_v2h3_mc.yaml --mode coldstart
```
