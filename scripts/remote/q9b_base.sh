#!/bin/bash
# 9B 对照的补充评测（单独成脚本，不改正在运行的 q9b_task.sh）：
#   base <9b|2b>   未训练基座（不挂 LoRA）在三套验收原文上生成，用评测所有 RAG 版本的同一个 9B 评测器打分
#                  → outputs/rt/core/q9b_base/eval/base<9|2>_{a,b,c}.jsonl / _9b.json（和各 GRPO 组的 eval 目录同格式）
#   cmp2           在 cmp 的基础上加两个基座组，配对比较 → results/q9b_cmp_full.json；
#                  两个 critic 的静态评测报告并排 → results/q9b_critic_static.json
cd ${CODE:-${FS_ROOT:-/root/autodl-fs}/code/critic_model}
A=${TMP_ROOT:-/root/autodl-tmp}; FS=${FS_ROOT:-/root/autodl-fs}; S=scripts/remote
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
log() { echo "$(date '+%F %T') q9b_base $*"; }
source $S/models9b.sh
B9=$A/models/Qwen3.5-9B
E=outputs/rt/core/q9b_base/eval; mkdir -p $E
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
temp() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }

case $1 in
base)
  sz=${2:?9b|2b}; m=base${sz%b}
  ensure_9b || exit 1
  if [ $sz = 9b ]; then MC=$B9; DUMMY=$FS/adapters/rag-sft-core_9b/adapter
  else source $S/model_env.sh minicpm5-2b || exit 1; MC=$MODEL_LOCAL; DUMMY=$FS/adapters/rag-sft-core/adapter; fi
  # serve_loras 至少要挂一个 LoRA；生成时请求 model=base，就是不挂 LoRA 的基座
  free_gpu
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-q9b-$m.log dummy=$DUMMY || exit 1
  python -u -m ragtruth.mc_accept_gen --ref outputs/rt/minicpm/grpo300/answers.jsonl --temperature 1.0 --top_p 0.95 --url http://127.0.0.1:8000/v1 --model base --out $E/${m}_a.jsonl &
  python -u -m ragtruth.mc_accept_gen --ref outputs/rt/minicpm/grpo300b/answers.jsonl --temperature 0.7 --top_p 0.95 --url http://127.0.0.1:8000/v1 --model base --out $E/${m}_b.jsonl &
  python -u -m ragtruth.mc_accept_gen --ref outputs/rt/minicpm/grpo300c/answers.jsonl --temperature 0.7 --top_p 0.95 --url http://127.0.0.1:8000/v1 --model base --out $E/${m}_c.jsonl &
  wait; free_gpu
  for s in a b c; do [ -s $E/${m}_$s.jsonl ] || { log "gen $m $s FAILED"; exit 1; }; done
  C9=rm1__rm1_gold_9b; T=$(temp $C9)
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $B9 --log $A/vllm-q9b-$m-score.log critic=$FS/adapters/$C9-9b/adapter || exit 1
  J="["; for s in a b c; do J+="{\"model\":\"critic\",\"temp\":$T,\"answers\":\"$E/${m}_$s.jsonl\",\"out\":\"$E/${m}_${s}_9b.json\"},"; done
  echo "${J%,}]" > $E/jobs_$m.json
  python -u -m ragtruth.critic_score_multi --url http://127.0.0.1:8000/v1 --jobs $E/jobs_$m.json --concurrency 256 || exit 1
  free_gpu
  log "BASE_DONE $m" ;;
cmp2)
  C=outputs/rt/core
  python -m ragtruth.pol9b_cmp --arm base2=$E:base2 --arm base9=$E:base9 \
    --arm sft2=$C/loop2_coreonly/eval:v0 --arm sft9=$C/loop2_q9b_p9r9/eval:v0 \
    --arm s1_2b=$C/loop2_coreonly/eval:s200 --arm s100_9b=$C/loop2_q9b_p9r9/eval:s100 --arm s200_9b=$C/loop2_q9b_p9r9/eval:s200 \
    --base s1_2b --n_boot 1000 --out results/q9b_cmp_full.json || exit 1
  python - <<'EOF' || exit 1
import json
out = {n: json.load(open(f"outputs/rt/rm/{n}/eval/report.json")) for n in ("rm1__core", "rm1__core_9b")}
json.dump(out, open("results/q9b_critic_static.json", "w"), ensure_ascii=False, indent=1)
EOF
  log "CMP2_DONE" ;;
*) echo "usage: q9b_base.sh base <9b|2b> | cmp2"; exit 1 ;;
esac
