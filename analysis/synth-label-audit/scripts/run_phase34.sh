#!/bin/bash
# Phase 3(critic 增广 A/B 建数据)+ Phase 4(重训 critic + rm_eval 出报告)
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-fs/code/critic_model || exit 1
export PYTHONHASHSEED=0 OPENAI_API_KEY=x
A=/root/autodl-tmp; HV=harvest_gated.jsonl
L() { echo "$(date '+%F %T') $*"; }
[ -s $HV ] || { echo "NO_HARVEST $HV"; exit 1; }
pkill -9 -f vllm 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null; sleep 5
for i in $(seq 1 40); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "$m" -lt 1500 ] && break; sleep 3; done
L "harvest lines: $(wc -l < $HV)"
# Phase 3: Aug-A 只负样本 / Aug-B 负+真·干净原句(1:1 平衡)。盲区带 [0,0.5)(critic 漏判),sev 真错门 0.5
L "build Aug-A (no_orig)"
python -m ragtruth.critic_harvest --harvest $HV --base rm1__core --name rm1__core_gA --lo 0 --hi 0.5 --gate 0.5 --min_nat 0 --max_add 2000 --per_prompt 2 --no_orig 2>&1 | tail -1
L "build Aug-B (paired orig)"
python -m ragtruth.critic_harvest --harvest $HV --base rm1__core --name rm1__core_gB --lo 0 --hi 0.5 --gate 0.5 --min_nat 0 --max_add 2000 --per_prompt 2 2>&1 | tail -1
# Phase 4: 重训 + 评测(report.json + eval/test_scores.jsonl)
for RM in rm1__core_gA rm1__core_gB; do
  L "train+eval $RM"
  rm -rf /root/autodl-fs/runs/claims/mcrm_$RM 2>/dev/null
  RM=$RM GPU=0 PORT=8101 bash scripts/remote/run_mcrm.sh 2>&1 | tail -3
done
L "PHASE34_DONE"
