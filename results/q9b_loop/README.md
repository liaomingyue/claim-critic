# 全 9B 闭环两轮的结果文件

对应 [docs/tech_share_details.md](../../docs/tech_share_details.md) 的表 11–14 和 [docs/experiments.md](../../docs/experiments.md) §8.5。

## RAG 配对比较（金标 critic，三套验收原文，596 对）

| 文件 | 内容 |
|---|---|
| `rag_r1_cmp.json` | 第一轮：起点（9B GRPO 200 步）对第一轮 critic 当奖励的第 50 / 100 步 |
| `rag_r1_ctl_cmp.json` | 第一轮对照：起点、奖励不换训 50 步、新 critic 第 50 / 100 步 |
| `rag_r2_cmp.json` | 第二轮：第一轮第 100 步对第二轮 critic（不配原句）当奖励的第 50 / 100 步 |

## Claude 对 RAG 回答的逐句盲标（`claude_rag/`）

每个目录里的 `final_<组>.jsonl` 是逐句标注：是否错误、严重程度、机制、证据、说明。`strat_info.json` 是分层抽样的层和权重：每个配对按所在层的权重（层内总数 ÷ 抽到的数）加权回总体。

| 目录 | 组 | 抽样 |
|---|---|---|
| `cof9/` | `s200` = 起点，`s100` = 第一轮第 100 步 | 120 对，按金标分数分层（高 60 / 中 36 / 低 24） |
| `dr2n/` | `d50`、`d100` = 第二轮第 50 / 100 步 | 从 `cof9/` 的 120 对里随机抽 100 对（`pairs100.json`），权重沿用 `cof9/strat_info.json` |
| `d50b/` | `c` = 第一轮第 100 步，`d50` = 第二轮第 50 步 | 新抽 100 对，和 `cof9/` 的 120 对不重叠（高 24 / 中 56 / 低 20） |

## Claude 对合成样本的核实（`claude_synth/`）

| 文件 | 内容 |
|---|---|
| `ev_s1c9b_zone.jsonl` … `ev_s4c9_zone.jsonl` | 第一轮四个阶段的生成器，在留出回答上的盲区样本，每段约 100 条 |
| `fill_of9s.jsonl` | 第一轮补进 critic 的 1,181 条里新抽审的 95 条（另有 37 条在各阶段核实里已标过，合计 132 条） |
| `fill_of9r2.jsonl` | 第二轮补进 critic 的 338 条全审，返回 335 条 |

## critic 在 9B 回答上的逐句分数

`critic_scores_9b_answers.json`：两个 9B RAG 策略（`q9b_p9r9`，就是第一轮起点；以及 `q9b_p9r2`）的 213 条回答，共 675 句、62 个错。每句带 Claude 标签，以及 9B 起点 critic（`rm1__core_r2_bfcl_9b`）和第一轮 critic（`rm1__core_of9all_9b`）的打分。
