# 合成"盲区错"标签审计 + gated/add 生成器实验（2026-11-08）

## 一句话结论
"骗 critic 造错 → 增广 critic"这条路线**走不通**：gated 生成器挑出的"盲区错"里
**只有 24% 是真错，76% 其实 faithful**。用它增广 critic 提不了 AUC、同误报率下反降。
根因是 reward hacking（弱 2B sev 门 + 盲区最大化双重击穿），不是过滤难度问题。

## 实验链（machine1 gated / machine2 add，均已跑完后关机）
1. **gated GRPO** 生成器（`run_gcore.sh`，`--shape gated`）→ harvest（`run_harvest_gated.sh`）
   挑出 band=[0,0.5)、gate=0.5、min_nat=0 的样本：**4376 条真错池，选 987 条"盲区错"**。
2. **critic 增广 A/B**（`run_phase34.sh`）：
   - Aug-A = 纯负样本注入（`critic_harvest.py --no_orig`），train 5591
   - Aug-B = 负+原句 1:1 平衡，train 6578
3. **重训 critic + 测盲区召回**（`blindrecall.py` / `matchfp.py`）。
4. **add GRPO** 生成器（machine2，`run_gadd_m2.sh`，`--shape add` WH2=0.6 WR2=0.3）跑完，
   因同弱门预计同样失败，未串 harvest/增广。

## 决定性诊断：Claude 审 200 条合成错（`judge.py` + `verify.py`）
- 复用本地 `claude -p --model opus --output-format json`，每批 10 条、独立判 faithful/conflict/baseless。
- 结果（`results/verdicts.jsonl`，n=200）：
  - **错句**（我们标成"错"的）：faithful **152**、conflict 30、baseless 18 → **真错仅 48/200 = 24%**。
  - 配对**原句**：faithful 184/200 = **92%**（说明 Claude 校准合理，不是它太严）。
  - 真错率 vs 盲区深度**单调反相关**：p_halluc [0,0.1)→14%、[0.1,0.3)→20%、[0.3,0.5)→38%。
    critic 漏判越狠的，越可能本来就没错。
- **核验兜底**（`verify.py`，`results/verify_res.json`）：把生成器自己给的 type/evidence/reason
  一并喂给 Claude 复核那 152 条 faithful → **只翻了 2 条（1.3%）**，仍 150 faithful。
  → 不是"漏看证据判错了"，是生成器**真错产出率本身只有 ~25%**。

## critic 增广结果（同误报率口径，`matchfp.py`）
| critic | 整体召回 | 盲区召回 | Claude金标 AUC |
| --- | --- | --- | --- |
| 基线 rm1__core | 0.555 | — | 0.830 |
| Aug-A（+987 盲区错） | 0.522 | 15.6% | 0.810 |
| Aug-B（+987 负+原句） | 0.516 | 15.6% | 0.810 |

两组都 < 基线（在教 critic 把 3/4 正常句报成错）。注意：固定阈值 recall@0.5 会**虚涨**（看起来"召回 45% 盲区"），
同误报率下才看清是阈值漂移假象。

## 脚本（`scripts/`）
- `run_gcore.sh` / `run_gadd_m2.sh` — gated / add 生成器 GRPO（远端）
- `run_harvest_gated.sh` — harvest 挑盲区错
- `run_phase34.sh` — 建 Aug-A/B + 重训 critic
- `blindrecall.py` — 盲区召回（baseline 漏的真错里 Aug 救回多少）
- `matchfp.py` — 同误报率对齐后的整体/盲区召回（防阈值漂移骗）
- `extract200.py` / `extract_ev.py` — 复现 gA 过滤 + seed42 抽 200（+evidence/reason）
- `judge.py` — Claude 独立判 faithful/conflict/baseless（批 10）
- `verify.py` — 给 Claude 生成器的 evidence/reason 做兜底核验
- `run_9b200.sh` — 9B holistic 给 200 条打分（关机前被杀，未产出，可重跑）

## 下一步方向（用户定调）
放弃合成造错。转**三块判别器**（证据判别 / 错误类型判别 / 理由判别）当地基：
既做造错的 GRPO 奖励（治 ~25% 真错产出率），也做挖真实错的标注器（更稳、不被钻）。
第一步：Claude 在这 200 条上出三块结构化标签，看哪块判别力最强 + 当判别器首批训练数据。

详见 `docs/experiment_log.md` 对应日期段 与 memory `claim-critic-synth-label-noise`。
