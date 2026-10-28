#!/bin/bash
# 本地一条龙：开机（或克隆）→ 上传最新代码 → 在 5090 上跑格式对比实验，接着跑合成错误样本实验 → 盯到结束
# → 下载结果（outputs/autodl/keep_on 存在时不关机，否则关机）。
# 在 Git Bash 里运行；MSYS_NO_PATHCONV 防止 /root/... 被改写成 Windows 路径。
set -u
cd /i/code/critic_model
export MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python
$PY -u -m scripts.autodl_poweron --uuid "${1:-611b489072-8df08a9e}" --clone_after_min "${CLONE_AFTER_MIN:-20}" || exit 1
for i in $(seq 1 30); do $PY -m scripts.remote run "echo ok" >/dev/null 2>&1 && break; sleep 20; done
D=/root/autodl-tmp/critic_model
for f in ragtruth/task.py ragtruth/resp.py ragtruth/eval.py ragtruth/metrics.py ragtruth/build.py ragtruth/annotate.py \
         ragtruth/synth.py ragtruth/mine.py common/llm.py \
         train/sft.py tests/test_ragtruth.py tests/test_rt_synth.py scripts/remote/run_format_exp.sh scripts/remote/run_synth_exp.sh \
         configs/rt/exp/sft_0.6b_e2.yaml configs/rt/exp/sft_0.6b_e3.yaml configs/rt/exp/sft_0.6b_e5.yaml configs/rt/exp/sft_0.6b_e5b.yaml \
         configs/rt/exp/sft_0.6b_syn_direct.yaml configs/rt/exp/sft_0.6b_syn_critic.yaml configs/rt/exp/sft_0.6b_syn_random.yaml \
         outputs/rt/claims/train_sub35_nat.jsonl outputs/rt/resp/train_full.jsonl outputs/rt/annotated/train_sub35.jsonl \
         outputs/external/ragtruth/train_pool.jsonl; do
  $PY -m scripts.remote put "$f" "$D/$f"
done
$PY -m scripts.remote run "sed -i 's/\r$//' $D/scripts/remote/*.sh"   # Windows 写出的 CRLF 会让 bash 报错
$PY -m scripts.remote run --bg --log /root/autodl-tmp/exp_runner.log "bash $D/scripts/remote/run_format_exp.sh"
$PY -m scripts.remote run --bg --log /root/autodl-tmp/syn_runner.log "bash $D/scripts/remote/run_synth_exp.sh"
$PY -u -m scripts.autodl_watch --exp_log /root/autodl-tmp/syn.log --done SYN_DONE --proc run_synth_exp
