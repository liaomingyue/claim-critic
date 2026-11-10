#!/bin/bash
# A 上奖励对照跑完后：
#   ① 新打分器验证（16 个 critic 评测 GPU 只有 33%，要打满）：16 个 critic 一次挂上（MAX_LORAS 16），critic_score_multi 一起打 grpo300c，
#      和 C 上旧 critic_score 的结果逐句比（应一致），记耗时、GPU 利用率
#   ② 奖励对照独立评测（9B 不进奖励，只当评测标签）：v0（mc-init）/ R0–R3 第 50 步在 200 个验收原文-种子上生成（同 run_eval_mid：grpo300 T1.0、grpo300b T0.7，top_p 0.95），
#      9B 金标 critic（rm1__rm1_gold_9b）逐句打分 → outputs/rt/minicpm/rwexp/eval/
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/after.log; S=scripts/remote; MC=$A/models/MiniCPM5-2B; E=outputs/rt/minicpm/rwexp/eval; mkdir -p $E
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false
unset VLLM_API_KEY
temp() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
ad() { [ -f $1/policy/adapter_model.safetensors ] && echo $1/policy || echo $1; }
until grep -q POST_DONE $A/post.log 2>/dev/null; do sleep 30; done

# ① 新打分器
D=outputs/rt/minicpm/grpo300c; V=$A/multi_check; mkdir -p $V
CR=(rm1__mcgold rm1__mcgold_ref rm1__mcgold_h2 rm1__mcgold_h3 rm1__mcgold_h3ctl rm1__mcgold_h3v rm1__mcgold_h3ctlv rm1__mcgold_h3ctlu
    rm1__mcgold_ref_z1500 rm1__mcgold_ref_z6000 rm1__mcgold_ref_divhi rm1__mcgold_ref_divlo rm1__mcgold_ref_n1500 rm1__mcgold_ref_nohA rm1__mcgold_ref_noh rm1__mcgold_ref_nohs)
mods=(); jobs="["
for k in "${!CR[@]}"; do
  r=${CR[$k]}; mods+=("c$k=$FS/adapters/$r-mc/adapter")
  jobs+="{\"model\":\"c$k\",\"temp\":$(temp $r),\"answers\":\"$D/answers.jsonl\",\"out\":\"$V/$r.json\"},"
done
echo "${jobs%,}]" > $V/jobs.json
free_gpu; mkdir -p $A/s0
MAX_LORAS=16 A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --base $MC --log $A/vllm-multi.log "${mods[@]}" >> $LOG 2>&1 || { log "serve 16 critics FAILED"; }
( while sleep 5; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; done ) > $V/util.txt &
U=$!
t0=$(date +%s)
python -u -m ragtruth.critic_score_multi --url http://127.0.0.1:8000/v1 --jobs $V/jobs.json --concurrency ${CONC:-384} >> $LOG 2>&1 || log "multi FAILED"
kill $U
log "MULTI_CHECK $(( $(date +%s) - t0 )) s, GPU util mean $(awk '{s+=$1;n++} END {printf "%.0f", s/n}' $V/util.txt)% (n=$(wc -l < $V/util.txt))"
python - $D $V "${CR[@]}" <<'EOF' 2>&1 | tee -a $LOG
import json, sys
d, v, crs = sys.argv[1], sys.argv[2], sys.argv[3:]
worst = 0.0
for r in crs:
    old = f"{d}/critic2b_scores.json" if r == "rm1__mcgold" else f"{d}/critic_{r}.json"
    try:
        a = {x["id"]: x["p"] for x in json.load(open(old))}; b = {x["id"]: x["p"] for x in json.load(open(f"{v}/{r}.json"))}
    except OSError as e:
        print("MULTI_DIFF", r, "missing", e); continue
    diff = max(abs(a[k] - b[k]) for k in a)
    worst = max(worst, diff)
    print("MULTI_DIFF", r, "n", len(a), len(b), "max_abs_diff", round(diff, 5), "flip@0.5", sum((a[k] >= .5) != (b[k] >= .5) for k in a))
print("MULTI_DIFF worst", round(worst, 5))
EOF
free_gpu

# ② 奖励对照独立评测
POL="v0=$FS/adapters/mc-init R0=$(ad $FS/adapters/rwexp-R0/step50) R1=$(ad $FS/adapters/rwexp-R1/step50) R2=$(ad $FS/adapters/rwexp-R2/step50) R3=$(ad $FS/adapters/rwexp-R3/step50)"
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --base $MC --log $A/vllm-rwgen.log $POL >> $LOG 2>&1 || { log "serve policies FAILED"; exit 1; }
for m in v0 R0 R1 R2 R3; do
  python -u -m ragtruth.mc_accept_gen --ref outputs/rt/minicpm/grpo300/answers.jsonl --temperature 1.0 --top_p 0.95 --url http://127.0.0.1:8000/v1 --model $m --out $E/${m}_a.jsonl >> $LOG 2>&1 &
  python -u -m ragtruth.mc_accept_gen --ref outputs/rt/minicpm/grpo300b/answers.jsonl --temperature 0.7 --top_p 0.95 --url http://127.0.0.1:8000/v1 --model $m --out $E/${m}_b.jsonl >> $LOG 2>&1 &
done
wait
free_gpu
C9=rm1__rm1_gold_9b
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --base $FS/models/Qwen3.5-9B --log $A/vllm-rw9b.log critic=$FS/adapters/$C9-9b/adapter >> $LOG 2>&1 || { log "serve 9b FAILED"; exit 1; }
T=$(temp $C9); jobs="["
for m in v0 R0 R1 R2 R3; do for s in a b; do jobs+="{\"model\":\"critic\",\"temp\":$T,\"answers\":\"$E/${m}_$s.jsonl\",\"out\":\"$E/${m}_${s}_9b.json\"},"; done; done
echo "${jobs%,}]" > $E/jobs9b.json
python -u -m ragtruth.critic_score_multi --url http://127.0.0.1:8000/v1 --jobs $E/jobs9b.json --concurrency 256 >> $LOG 2>&1 || log "9b score FAILED"
free_gpu
log "AFTER_DONE"
