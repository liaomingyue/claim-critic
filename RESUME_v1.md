# v1 自举采样实验 — 断点（本地重启后接续用）

## GPU 实例（勿关机，在计费）
- 新实例: ADLH-1107155004 / uuid `676946bb52-888b9d8d`，西北D区 RTX 5090
- SSH 信息: `outputs/autodl/v1/web_instance.json`（proxy_host/ssh_port/root_password）
- 网页 token: `autodl_web.json`；Pro token: `outputs/autodl/autodl_token.txt`
- 930机（无卡 cbcf4b8f5f）也还开着，可关
- 没有看门狗在跑 → 不会自动关机（会持续计费 ~2.78/时）

## 已完成
- 2B + 9B 模型已拷到本地盘 `/root/autodl-tmp/models/`
- 9B 金标 critic vLLM 服务在 `http://127.0.0.1:8000/v1`（model=critic, 温度 1.3）仍在跑
- score9b 完成: 3161 严门样本 → **9B 翻转率 66.9%（2114 过）**；否掉的 89% 是「原句其实不干净」（sev 的 F 弱）
- 翻转门数据已建: `outputs/rt/rm/rm1__core_v1`（train 8626 = rm1__core 4604 + 2011 错 + 2011 原句）

## 下一步（回来继续）
1. 训 critic: 停 9B 服务腾显存 → `run_mcrm.sh` 训 `rm1__core_v1`（+ 对照组 CTL：同条数但不走翻转门）+ 基线 `rm1__core`
2. 评测: critic_cap3（Claude 金标 AUC）+ 被钻空子，vs rm1__core → 样本效率结论

## 本地助手（重启后 /tmp 会丢，需重建）
- `/tmp/ssh_run.exp host port pw cmd`（timeout 60）、`/tmp/ssh_long.exp`（timeout 300）、`/tmp/scp_get.exp`
- 远端命令要激活 conda: `bash -c "export PATH=/root/miniconda3/bin:\$PATH; ..."` 或 `bash -lc`
- 代码(含 score9b/翻转门)已部署远端 `/root/autodl-fs/code/critic_model`，也已 push GitHub loop-v1
