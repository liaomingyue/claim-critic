# 实验运行登记

由 `python -m scripts.collect_runs` 生成，每行对应一份 manifest（产出目录里的 run_manifest.json / eval_manifest.json）。复现时按 manifest 里的完整配置、数据 sha256 和代码指纹核对。

| 完成时间 | 类型 | 名称 | 数据 | 关键配置 | 耗时 | 结果 | 机器 |
|---|---|---|---|---|---|---|---|
| 补录 | run | sft-0.6b | train_sub.jsonl · 1410 行 · 814587df<br>sft_0.6b.yaml · ? 行 · e3a6ae39 | Qwen3-0.6B · style full · lr 0.0002 · ep 3 · bs 4×4 | — | — | local |
| 补录 | eval | sft-0.6b | test.jsonl · 2675 行 · 11da9fbe | style full · resp_style None · max_tokens None | — | 句 P/R/F1 0.537/0.578/0.557 · 回答 R 0.755 | local |
| 补录 | run | sft-0.6b-claim | train_sub.jsonl · 3003 行 · 77f9e3b5<br>sft_0.6b.yaml · ? 行 · e3a6ae39 | Qwen3-0.6B · style full · lr 0.0002 · ep 3 · bs 4×4 | — | — | local |
| 补录 | eval | sft-0.6b-claim | test.jsonl · 2675 行 · 11da9fbe | style full · resp_style None · max_tokens None | — | 句 P/R/F1 0.314/0.706/0.434 · 回答 R 0.889 | local |
| 补录 | run | sft-0.6b-v2-notips | train_sub35_r3.jsonl · 3484 行 · d95fb858<br>sft_0.6b_v2_notips.yaml · ? 行 · ffd39871 | Qwen3-0.6B · style full · lr 0.0002 · ep 3 · bs 4×4 | — | — | local |
| 补录 | eval | sft-0.6b-v2-notips | test.jsonl · 2675 行 · 11da9fbe | style full · resp_style None · max_tokens None | — | 句 P/R/F1 0.340/0.717/0.461 · 回答 R 0.880 | local |
| 补录 | run | sft-9b-claim | train_sub.jsonl · 3003 行 · 77f9e3b5<br>sft_9b.yaml · ? 行 · 1e358035 | Qwen3.5-9B · style full · lr 0.0001 · ep 2 · bs 1×16 | 49.9 分钟 | loss 0.1204 · 376 步 | 611b489072（已释放 |
| 补录 | eval | sft-9b-claim | test.jsonl · 2675 行 · 11da9fbe | style full · resp_style None · max_tokens None | — | 句 P/R/F1 0.470/0.816/0.596 · 回答 R 0.922 | 611b489072（已释放 |
| 补录 | run | sft-9b-v2-notips | train_sub35_r3.jsonl · 3484 行 · d95fb858<br>sft_9b_v2.yaml · ? 行 · 492205ee | Qwen3.5-9B · style full · lr 0.0001 · ep 2 · bs 1×16 | 56.4 分钟 | loss 0.09915 · 436 步 | 611b489072（已释放 |
| 补录 | eval | sft-9b-v2-notips | test.jsonl · 2675 行 · 11da9fbe | style full · resp_style None · max_tokens None | — | 句 P/R/F1 0.508/0.803/0.623 · 回答 R 0.906 | 611b489072（已释放 |
| 补录 | run | sft-0.6b-e2 | train_sub35_nat.jsonl · 8298 行 · a394904b<br>sft_0.6b_e2.yaml · ? 行 · a7f4b945 | Qwen3-0.6B · style verdict · lr 0.0002 · ep 3 · bs 8×2 | 61.5 分钟 | loss 0.003178 · 1557 步 | 65ee479064（已释放 |
| 补录 | eval | sft-0.6b-e2 | test.jsonl · 2675 行 · 11da9fbe | style verdict · resp_style None · max_tokens None | — | 句 P/R/F1 0.643/0.617/0.630 · 回答 R 0.739 | 65ee479064（已释放 |
| 补录 | run | sft-0.6b-e3 | train_sub35_nat.jsonl · 8298 行 · a394904b<br>sft_0.6b_e3.yaml · ? 行 · a18333d6 | Qwen3-0.6B · style full · lr 0.0002 · ep 3 · bs 8×2 | 73.8 分钟 | loss 0.1255 · 1557 步 | 65ee479064（已释放 |
| 补录 | eval | sft-0.6b-e3 | test.jsonl · 2675 行 · 11da9fbe | style full · resp_style None · max_tokens None | — | 句 P/R/F1 0.515/0.582/0.547 · 回答 R 0.738 | 65ee479064（已释放 |
| 补录 | run | sft-0.6b-e5 | train_full.jsonl · 774 行 · 80f93cc2<br>sft_0.6b_e5.yaml · ? 行 · f88f5b18 | Qwen3-0.6B · style evidence · lr 0.0002 · ep 3 · bs 8×2 | — | 147 步 | c5k0nax3ry（主实例 |
| 补录 | eval | sft-0.6b-e5 | test.jsonl · 2675 行 · 11da9fbe | style evidence · resp_style evidence/plain · max_tokens None | — | 句 P/R/F1 0.257/0.432/0.323 · 回答 R 0.696 | c5k0nax3ry（主实例 |
| 补录 | run | sft-0.6b-e5b | train_full.jsonl · 774 行 · 80f93cc2<br>sft_0.6b_e5b.yaml · ? 行 · 56e86db7 | Qwen3-0.6B · style plain · lr 0.0002 · ep 3 · bs 8×2 | — | 147 步 | c5k0nax3ry（主实例 |
| 补录 | eval | sft-0.6b-e5b | test.jsonl · 2675 行 · 11da9fbe | style plain · resp_style evidence/plain · max_tokens None | — | 句 P/R/F1 0.323/0.611/0.423 · 回答 R 0.837 | c5k0nax3ry（主实例 |
| 补录 | run | sft-0.6b-e4 | train_sub35_nat.jsonl · 8298 行 · a394904b<br>sft_0.6b_e4.yaml · ? 行 · 9d7f67ab | Qwen3-0.6B · style vfirst · lr 0.0002 · ep 3 · bs 8×2 | 72.2 分钟 | loss 0.1299 · 1557 步 | gzw4qpf8tl（机器  |
| 补录 | eval | sft-0.6b-e4 | test.jsonl · 2675 行 · 11da9fbe | style vfirst · resp_style None · max_tokens None | — | 句 P/R/F1 0.633/0.563/0.596 · 回答 R 0.695 | gzw4qpf8tl（机器  |
| 补录 | run | sft-9b-verdict | train_sub35_r3.jsonl · 3484 行 · d95fb858<br>sft_9b_verdict.yaml · ? 行 · 0eacfefa | Qwen3.5-9B · style verdict · lr 0.0001 · ep 2 · bs 1×16 | 49.0 分钟 | loss 0.007005 · 436 步 | gzw4qpf8tl（机器  |
| 补录 | eval | sft-9b-verdict | test.jsonl · 2675 行 · 11da9fbe | style verdict · resp_style None · max_tokens None | — | 句 P/R/F1 0.599/0.806/0.687 · 回答 R 0.888 | gzw4qpf8tl（机器  |
| 补录 | run | sft-0.6b-syn-direct | syn_r1_direct.jsonl · 11972 行 · 9d7b87ca<br>sft_0.6b_syn_direct.yaml · ? 行 · 4b229b2b | Qwen3-0.6B · style full · lr 0.0002 · ep 3 · bs 8×2 | 100.7 分钟 | loss 0.1414 · 2247 步 | c5k0nax3ry（主实例 |
| 补录 | eval | sft-0.6b-syn-direct | test.jsonl · 2675 行 · 11da9fbe | style full · resp_style None · max_tokens None | — | 句 P/R/F1 0.387/0.655/0.486 · 回答 R 0.831 | c5k0nax3ry（主实例 |
| 补录 | run | sft-0.6b-syn-critic | syn_r1_critic.jsonl · 10730 行 · 95f02d4b<br>sft_0.6b_syn_critic.yaml · ? 行 · fc2ebf27 | Qwen3-0.6B · style full · lr 0.0002 · ep 3 · bs 8×2 | 91.4 分钟 | loss 0.1246 · 2013 步 | c5k0nax3ry（主实例 |
| 补录 | eval | sft-0.6b-syn-critic | test.jsonl · 2675 行 · 11da9fbe | style full · resp_style None · max_tokens None | — | 句 P/R/F1 0.373/0.656/0.475 · 回答 R 0.844 | c5k0nax3ry（主实例 |
| 补录 | run | sft-0.6b-syn-random | syn_r1_random.jsonl · 10730 行 · 0dae85f2<br>sft_0.6b_syn_random.yaml · ? 行 · 8c5e56f3 | Qwen3-0.6B · style full · lr 0.0002 · ep 3 · bs 8×2 | 91.0 分钟 | loss 0.1189 · 2013 步 | c5k0nax3ry（主实例 |
| 补录 | eval | sft-0.6b-syn-random | test.jsonl · 2675 行 · 11da9fbe | style full · resp_style None · max_tokens None | — | 句 P/R/F1 0.407/0.648/0.500 · 回答 R 0.830 | c5k0nax3ry（主实例 |
| 补录 | run | sft-9b-syn-direct | syn9b_r1_direct.jsonl · 7158 行 · 95f198cc<br>sft_9b_syn_direct.yaml · ? 行 · 66bfdfe9 | Qwen3.5-9B · style full · lr 0.0001 · ep 2 · bs 1×16 | 108.3 分钟 | loss 0.122 · 896 步 | gzw4qpf8tl（机器  |
| 补录 | eval | sft-9b-syn-direct | test.jsonl · 2675 行 · 11da9fbe | style full · resp_style None · max_tokens None | — | 句 P/R/F1 0.459/0.821/0.589 · 回答 R 0.932 | gzw4qpf8tl（机器  |
| 补录 | run | sft-9b-syn-critic | syn9b_r1_critic.jsonl · 4404 行 · 8bb8bad5<br>sft_9b_syn_critic.yaml · ? 行 · e22118a9 | Qwen3.5-9B · style full · lr 0.0001 · ep 2 · bs 1×16 | 69.2 分钟 | loss 0.122 · 552 步 | pwr6zxz5jh（机器  |
| 补录 | eval | sft-9b-syn-critic | test.jsonl · 2675 行 · 11da9fbe | style full · resp_style None · max_tokens None | — | 句 P/R/F1 0.414/0.838/0.554 · 回答 R 0.953 | pwr6zxz5jh（机器  |
| 补录 | run | sft-9b-syn-random | syn9b_r1_random.jsonl · 4404 行 · 3bf20d3d<br>sft_9b_syn_random.yaml · ? 行 · 119be769 | Qwen3.5-9B · style full · lr 0.0001 · ep 2 · bs 1×16 | 67.8 分钟 | loss 0.1288 · 552 步 | pwr6zxz5jh（机器  |
| 补录 | eval | sft-9b-syn-random | test.jsonl · 2675 行 · 11da9fbe | style full · resp_style None · max_tokens None | — | 句 P/R/F1 0.479/0.843/0.611 · 回答 R 0.927 | pwr6zxz5jh（机器  |
| 补录 | eval | zs-qwen3-0.6b | test.jsonl · 2675 行 · 11da9fbe | style None · resp_style None · max_tokens None | — | 句 P/R/F1 0.000/0.000/0.000 · 回答 R 0.000 | local |
