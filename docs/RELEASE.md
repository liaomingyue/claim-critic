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

每个模型用的训练数据（具体文件和条数）、训练配置（超参文件、LoRA、学习率、步数、奖励）、起点、重训命令，以及它对应哪张表，都列在 [MODELS.md](MODELS.md)。发布包里每个模型目录下的 `README.md` 和 `training.json` 也是同样的内容。

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
pip install -r requirements.txt
python -m scripts.fetch_release --data_repo <用户>/claim-critic-data --only data --parts eval
C="rm1__core rm1__core_r2 rm1__core_r2_bfcl rm1__core_r2_bfnc rm1__core_rd2"
E=outputs/rt/core
# critic：多类型补盲核实 vs 不核实（表 7），以不核实组为基线
python -m ragtruth.critic_cap3 --critics $C --base rm1__core_r2_bfnc --by_type --out r_clnc.json
python -m scripts.check_repro r_clnc.json results/critic_cap_blindfill_clnc.json
# critic：第二轮 rd2 vs 第一轮 bfcl 的静态评测（表 9）
python -m ragtruth.critic_cap3 --critics $C --base rm1__core_r2_bfcl --by_type --out r_rd2.json
python -m scripts.check_repro r_rd2.json results/critic_cap_rd2.json
# RAG：第一轮闭环 + 对照、第二轮（表 8、9）；配对集合由参与的版本共同决定，所以版本列表要和原来一致
python -m ragtruth.pol9b_cmp --arm sft=$E/loop2_coreonly/eval:v0 --arm s1=$E/loop2_coreonly/eval:s200     --arm s2=$E/loop2_cont/eval:s100 --arm s3r1=$E/loop2_cont5/eval:s100 --arm bf_v0=$E/loop2_cont4/eval:v0     --arm s3bf=$E/loop2_cont4/eval:s100 --arm s4_v0=$E/loop2_cont6/eval:v0 --arm s4_50=$E/loop2_cont6/eval:s50     --arm s4_100=$E/loop2_cont6/eval:s100 --base s3bf --n_boot 1000 --out r_s4.json
python -m scripts.check_repro r_s4.json results/loop2_cont6_cmp.json
```

这一层从 Claude 金标、critic 逐句打分和 9B 逐句打分重新算 AUC、召回和配对 bootstrap 区间。种子固定，所以结果应和 `results/` 里的逐位一致，`check_repro` 会打印 `REPRO_MATCH`。

我们在一个干净目录里实际走过一遍：
- 从仓库 clone；
- 用 `fetch_release --local` 解压全部 9 个包，SHA-256 全部一致；
- 跑 L0；
- 跑上面三条 L1。

结果是三个文件都逐字段一致（含 bootstrap 区间）。注意两个容易踩的坑：
- `--base` 不同，差值列就不同；
- `pol9b_cmp` 只比所有版本都有回答的那些题，少放一个版本，配对条数就可能从 595 变成 596，比例会差零点几个点。

### 在 GPU 服务器上重跑（scripts/repro）

这一层不再拿我们算好的分数，而是在你自己的机器上用发布的 LoRA 重新打分、重新训练。需要：
- Linux 和 1 张显存不少于 32 GB 的卡，我们用的是 RTX 5090；
- `pip install -r requirements-gpu.txt`；
- 约 60 GB 磁盘。

```bash
git clone https://github.com/liaomingyue/claim-critic && cd claim-critic
bash scripts/repro/setup.sh --data_repo <用户>/claim-critic-data --model_repo <用户>/claim-critic-models
bash scripts/repro/r1_rescore.sh        # R1：约半小时
bash scripts/repro/r2_rag.sh            # R2：两组各约 6–8 小时；两台机器可以各跑一组：r2_rag.sh rp_bf / rp_r1，最后 r2_rag.sh cmp
```

`setup.sh` 做四件事：
1. 检查 GPU 和 vLLM 等依赖；
2. 下载并校验发布包；
3. 准备基座模型：MiniCPM5-2B 固定在提交 f974000，另加 Qwen3.5-9B；如果机器上已经有这两个模型，可以用 `--base_src <目录>` 直接拷过来；
4. 把 LoRA 按训练脚本要找的目录结构做成软链接。

所有东西都放在仓库下的 `.repro/` 里，可以用 `FS_ROOT` / `TMP_ROOT` 改到别的盘。原实验跑在 AutoDL 上，`scripts/remote/` 里的脚本在不设这两个变量时仍然用 AutoDL 的路径。

**R1 重新打分**：用 6 代 critic 重新给三套验收回答逐句打分。分数写到 `critic_<名字>_re.json`，不覆盖发布的分数。报告分三部分：
- 同一句话两次打分差多少；
- 每个 critic 的 AUC 变化；
- 两条结论在新分数上是否成立：
  - 补盲时用 Claude 核实比不核实好（表 7）；
  - 第二轮 critic 的静态指标反而变差（表 9）。

全部成立时打印 `R1_CONCLUSIONS_HOLD`。

**R2 重训 RAG**：复现核心结论"闭环的提升来自新 critic"（表 8）。从发布的 RAG-S2 出发训两组，各 100 步，只有奖励里的 critic 不同：
- `rp_bf`：用第一轮补盲后的 critic；
- `rp_r1`：用原来的 critic。

训完在三套验收原文上生成，用 9B 金标 critic 打分，再做配对 bootstrap。判据：新 critic 组比对照组的含显然错、含任意错比例都更低，且 90% 区间不含 0。成立时打印 `R2_CONCLUSION_HOLDS`。

GRPO 的采样和生成用 vLLM，有非确定性，所以 R2 的数字不会和原表逐位一致。举个例子：同一个起点模型重新生成一次，含显然错的比例就会在 6.5% 和 7.5% 之间变动。所以 R2 看的是结论方向和显著性，不看小数点。

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
