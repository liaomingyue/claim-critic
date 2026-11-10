#!/bin/bash
# 推理机空闲时用严重程度 RM 给 harvest h1 过门槛样本打分（推理机空闲别浪费）。临时挂 sev LoRA、低优先级打分、打完卸载。
set -u
cd /root/autodl-fs/code/critic_model
FS=/root/autodl-fs; A=/root/autodl-tmp; LOG=$A/sev_score.log
export VLLM_API_KEY=$(cat $FS/runs/pair/$(cat $FS/runs/pair/CURRENT 2>/dev/null || echo default)/key) PYTHONHASHSEED=0
H=(-H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json")
mkdir -p $A/lora && rm -rf $A/lora/sev && python scripts/strip_visual_lora.py $FS/adapters/sev__mc-mc/adapter $A/lora/sev >> $LOG 2>&1
curl -s "${H[@]}" -X POST http://127.0.0.1:6006/v1/load_lora_adapter -d "{\"lora_name\": \"sev\", \"lora_path\": \"$A/lora/sev\"}" >> $LOG 2>&1; echo >> $LOG
T=$(python -c "import json;print(json.load(open('outputs/rt/rm/sev__mc/eval/report.json'))['temperature'])")
python -u -m ragtruth.sev_score --url http://127.0.0.1:6006/v1 --model sev --temp $T --out outputs/rt/gentarget/sev_h1.jsonl >> $LOG 2>&1
curl -s "${H[@]}" -X POST http://127.0.0.1:6006/v1/unload_lora_adapter -d '{"lora_name": "sev"}' >> $LOG 2>&1
echo "SEV_SCORE_DONE" >> $LOG
