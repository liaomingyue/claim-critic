# claim-critic（RAGTruth 主线，loop-v1）

RAG 回答的逐句幻觉检测 critic，加一个"造边界错误 → 校验 → 加入 critic 训练集 → 用 critic 做奖励训练回答模型"的循环。全部训练用的模型都是 MiniCPM5-2B（下称 2B）；Qwen3.5-9B（下称 9B）只用于不进训练的评测和监控；Claude 只用于标注和验收。

- 结果与结论：[docs/results.md](docs/results.md)（按组件整理，带关键数字）
- 完整实验记录：[docs/experiment_log.md](docs/experiment_log.md)（按时间顺序，含命令、踩坑）
- 金标标注规范：[docs/gold_standard.md](docs/gold_standard.md)；生成器 GRPO 设计：[docs/gen_grpo.md](docs/gen_grpo.md)

本分支（`loop-v1`）只保留当前循环要用的代码。早期实验（中文医学 claim、9B 时代的 DPO / 闭环、各种探针）的完整代码在本地分支 `archive/full-20261104` 和本地全量备份里，不推送。

## 循环

```
错误生成器（2B，GRPO）在训练原文上改写回答里的一句，造当前 critic 拿不准 / 判不出的错误
  → 校验：严重程度 RM（sev，6 选 1）/ RM3 判真错且显然，RM2 判读起来自然，原句本身干净
  → 加入 critic（RM1）训练集，配同位置原句作忠实样本，重训 critic
  → critic 作为回答 GRPO 的奖励（加满意度护栏、拒答 RM、硬规则）训练 2B 回答模型
  → 验收：9B 金标 critic 补打 + Claude 按金标规范标注（同原文、同种子配对）
```

所有 RM 都是"单字母选项分类"：只读答案位置各选项的 logprob，在校准集上拟合一个全局温度，输出校准后的概率。

## 目录

| 位置 | 内容 |
|---|---|
| `common/` | 样本格式与 JSONL 读写（`schema`）、模型调用封装（`llm`：OpenAI 兼容 / Anthropic / Claude CLI）、prompt、运行清单 |
| `ragtruth/task.py`、`convert.py`、`resp.py`、`synth.py` | RAGTruth 逐句 critic 任务格式、原始数据转换、合成错误句的基础工具 |
| `ragtruth/rm_data.py`、`rm_eval.py` | 各 RM 的单字母数据构建（系统提示、选项）和打分 / 校准 / 评测 |
| `ragtruth/critic_aug.py` | 基线 critic 数据（`rm1__mcgold`：RAGTruth 人工 + 9B 回答 Claude 金标） |
| `ragtruth/critic_harvest.py`、`critic_refuse.py` | critic 加边界样本（h1 / h2）、拒答标签修正（ref） |
| `ragtruth/critic_score.py`、`critic_types.py`、`evidence_probe.py`、`attr_probe.py` | critic 在验收回答上逐句打分、按错误类型统计、漏检诊断探针 |
| `ragtruth/rm3_v2.py`、`audit_synth.py`、`b_check.py` | RM3（合成错误是否真错）的重训、渲染、Claude 复核 |
| `ragtruth/sev_label.py`、`sev_rm.py` | 严重程度：Claude 分字段标注（显然 / 非显然 / 没有错误 + 解释对不对）→ 6 选 1 RM 数据、门槛评测 |
| `ragtruth/gen_target_data.py`、`gen_indep_eval.py`、`boundary_metric.py`、`gen_eval.py` | 定向造错生成器的 prompt 池、种子、批量造数据（harvest）和评测 |
| `ragtruth/discrim.py` | RM2（自然 / 合成）输入格式 |
| `ragtruth/ans_data.py`、`ans_accept.py`、`ans_quality.py`、`accept_compare.py`、`accept_more.py`、`mc_accept_gen.py`、`shadow_rescore.py` | 回答 GRPO 的 prompt 池、Claude 验收、质量抽查、9B 补打 |
| `ragtruth/refuse_data.py`、`sat_data.py`、`sat_build.py`、`gold.py`、`pp_valid.py` | 拒答 RM、满意度 RM、金标集、RAGTruth++ 有效子集的数据 |
| `train/sft.py` | SFT / 单字母 RM 训练（`--mode coldstart`） |
| `train/gen_grpo.py`、`gen_reward.py` | 错误生成器 GRPO 和奖励 |
| `train/ans_grpo.py`、`ans_reward.py`、`prefix_share_llama.py` | 回答 GRPO、奖励、训练侧 prompt 共享 |
| `scripts/autodl_*.py`、`remote.py` | AutoDL：抢卡建实例（`autodl_extra --fs --roles <角色>`）、开机、释放、远程执行 |
| `scripts/remote/` | 远端流程脚本：`bootstrap_fs.sh`（按角色起步）、`serve_loras.sh`（多 LoRA vLLM，必须用它）、`run_mcrm.sh`（训一个 2B 单字母 RM）、`run_mcgrpo.sh`（回答 GRPO）、`run_rewardexp9.sh`（生成器 GRPO）、`run_harvest.sh` / `run_clean.sh` / `run_refuse.sh`（造数据和 critic 重训）、`run_sev.sh`（严重程度 RM）、`run_grpo_h2.sh`、`run_shadow9b.sh`、`run_bigeval.sh`、`run_gentarget3.sh` |
| `configs/` | RM 训练超参（`rt/rm_mc.yaml`）、生成器种子 SFT 配置、`data/`（Claude 标签：`sev_labels.jsonl`、`rm3v2_labels.jsonl`） |
| `tests/` | 单测（不需要 GPU） |

## 本地验证

```bash
pip install -e ".[dev]"
PYTHONHASHSEED=0 python -m pytest -q        # 全部单测，不需要 GPU、不调用模型
```

## 远端运行（AutoDL，文件存储模式）

数据、模型、adapter 都在文件存储 `/root/autodl-fs`（`code/critic_model/outputs/rt/...`、`adapters/<名字>-mc/`、`models/`）。本地只放代码；登录信息文件（`autodl_web.json`、`autodl_token.txt`、`smtp.json`）不进仓库。

```bash
python -m scripts.autodl_extra --fs --roles sevrm --gpus 1 --gpu "RTX 5090"     # 抢一台单卡跑严重程度 RM
python -m scripts.autodl_extra --fs --roles grpoh2 --gpus 2 --gpu "RTX 5090"    # 抢一台双卡跑回答 GRPO
```

角色和对应脚本见 `scripts/remote/bootstrap_fs.sh`。所有运行固定 `PYTHONHASHSEED=0` 和随机种子，重跑要能得到同样的数据。

## 当前状态（2026-11-04）

- critic：基线 `rm1__mcgold`；边界样本版 `rm1__mcgold_h2`；拒答修正版 `rm1__mcgold_ref`
- 严重程度 RM：`sev__mc`（显然 vs 其余 AUC 0.805；critic 盲区独立测试 0.714，门槛 0.5 时入库样本约 93% 确实有错）
- 下一步：回答 GRPO 用 h2 critic 当奖励（`run_grpo_h2.sh`）；新生成器 GRPO 以 sev RM P(显然) ≥ 0.5 为门槛、奖励 1 − P_critic；造数据 → critic 重训 → 验收。详见 `docs/results.md` §12。
