#!/bin/bash
# SFT 上限 + critic 被钻空子程度。单卡，三段：
#   ① 被钻空子：三档第 200 步回答（和 v0）用各自的 2B critic 打分（训练时进奖励的那个），对比 9B → $X
#   ② SFT 加训：S1 同数据训 3 / 4 / 6 轮（各自从头，LR 调度随轮数）→ adapters/policy-s1e{3,4,6}
#   ③ 评测：三个 S1 变体在三套验收原文上生成；9B 一次打完：变体回答 + S1 训练目标（tgt）+ 原参考回答（orig）
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/sftcap.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch /root/autodl-tmp/KEEPALIVE; sleep 60; done ) & KA=$!
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
unset VLLM_API_KEY
[ -f $A/models/MiniCPM5-2B/config.json ] || { mkdir -p $A/models && rsync -a $FS/models/MiniCPM5-2B $A/models/; }
source scripts/remote/models9b.sh; start_9b_bg
MC=$A/models/MiniCPM5-2B; M=outputs/rt/minicpm; X=$M/r3cmp_exploit; O=$M/s1; E=$O/cap; mkdir -p $X $E
temp() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
waitjobs() { for p in $(jobs -p); do [ "$p" = "$KA" ] || wait $p; done; }
declare -A CR=([ref]=rm1__mcgold_ref [z1500]=rm1__mcgold_ref_z1500 [h3ctl]=rm1__mcgold_h3ctl)
declare -A DIR=([ref]=$M/r3full_ref/eval [z1500]=$M/r3full_z1500/eval [h3ctl]=$M/r3full/eval)
# ① 各档自己的 critic 打分
free_gpu; mkdir -p $A/s0
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --base $MC --log $A/vllm-cap2b.log \
  c_ref=$FS/adapters/rm1__mcgold_ref-mc/adapter c_z1500=$FS/adapters/rm1__mcgold_ref_z1500-mc/adapter c_h3ctl=$FS/adapters/rm1__mcgold_h3ctl-mc/adapter >> $LOG 2>&1 || { log "serve 2b FAILED"; exit 1; }
jobs="["; for t in ref z1500 h3ctl; do T=$(temp ${CR[$t]}); for s in a b c; do
  jobs+="{\"model\":\"c_$t\",\"temp\":$T,\"answers\":\"${DIR[$t]}/s200_$s.jsonl\",\"out\":\"$X/${t}_s200_$s.json\"},"
  jobs+="{\"model\":\"c_$t\",\"temp\":$T,\"answers\":\"$M/r3full/eval/v0_$s.jsonl\",\"out\":\"$X/${t}_v0_$s.json\"},"
done; done; echo "${jobs%,}]" > $X/jobs.json
python -u -m ragtruth.critic_score_multi --url http://127.0.0.1:8000/v1 --jobs $X/jobs.json --concurrency 384 >> $LOG 2>&1 || log "own-critic score FAILED"
free_gpu; log "exploit scoring done"
# ② S1 加训
python -u -m ragtruth.policy_sft s1_targets --out $O >> $LOG 2>&1 || log "s1_targets FAILED"
for e in 3 4 6; do
  n=policy-s1e$e; CK=$A/ckpt/$n
  [ -f $FS/adapters/$n/adapter/adapter_model.safetensors ] && continue
  sed -e "s|base_model: .*|base_model: $MC|" -e "s|data: .*|data: $O/sft_train.jsonl|" -e "s|output_dir: .*|output_dir: $CK|" -e "s|num_train_epochs: .*|num_train_epochs: $e|" configs/rt/sft_policy_mc.yaml > configs/rt/exp/$n.yaml
  python -u -m train.sft --config configs/rt/exp/$n.yaml --mode coldstart > $A/$n.log 2>&1
  [ -f $CK/adapter/adapter_model.safetensors ] || { log "$n FAILED"; continue; }
  mkdir -p $FS/adapters/$n && rsync -a $CK/adapter $FS/adapters/$n/ && log "$n done: $(grep -ao "'train_loss': '[0-9.]*'" $A/$n.log | tail -n 1)"
done
# ③ 生成 + 9B
POL=""; for e in 3 4 6; do [ -f $FS/adapters/policy-s1e$e/adapter/adapter_model.safetensors ] && POL+=" s1e$e=$FS/adapters/policy-s1e$e/adapter"; done; POL=${POL# }
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --base $MC --log $A/vllm-capgen.log $POL >> $LOG 2>&1 || { log "serve gen FAILED"; exit 1; }
for kv in $POL; do m=${kv%%=*}
  python -u -m ragtruth.mc_accept_gen --ref $M/grpo300/answers.jsonl --temperature 1.0 --top_p 0.95 --url http://127.0.0.1:8000/v1 --model $m --out $E/${m}_a.jsonl >> $LOG 2>&1 &
  python -u -m ragtruth.mc_accept_gen --ref $M/grpo300b/answers.jsonl --temperature 0.7 --top_p 0.95 --url http://127.0.0.1:8000/v1 --model $m --out $E/${m}_b.jsonl >> $LOG 2>&1 &
  python -u -m ragtruth.mc_accept_gen --ref $M/grpo300c/answers.jsonl --temperature 0.7 --top_p 0.95 --url http://127.0.0.1:8000/v1 --model $m --out $E/${m}_c.jsonl >> $LOG 2>&1 &
done
waitjobs; free_gpu
C9=rm1__rm1_gold_9b; T=$(temp $C9)
ensure_9b || { log "9b copy FAILED"; exit 1; }
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --base $A/models/Qwen3.5-9B --log $A/vllm-cap9b.log critic=$FS/adapters/$C9-9b/adapter >> $LOG 2>&1 || { log "serve 9b FAILED"; exit 1; }
jobs="["; for kv in $POL; do m=${kv%%=*}; for s in a b c; do jobs+="{\"model\":\"critic\",\"temp\":$T,\"answers\":\"$E/${m}_$s.jsonl\",\"out\":\"$E/${m}_${s}_9b.json\"},"; done; done
for k in tgt orig; do jobs+="{\"model\":\"critic\",\"temp\":$T,\"answers\":\"$E/${k}_t.jsonl\",\"out\":\"$E/${k}_t_9b.json\"},"; done
echo "${jobs%,}]" > $E/jobs9b.json
python -u -m ragtruth.critic_score_multi --url http://127.0.0.1:8000/v1 --jobs $E/jobs9b.json --concurrency 256 >> $LOG 2>&1 || log "9b score FAILED"
free_gpu
log "SFTCAP_DONE"
