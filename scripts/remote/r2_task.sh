#!/bin/bash
# 第二轮闭环的单个任务（由 queue_worker.sh 领取执行）：r2_task.sh <pool|samp_v2h3|samp_g1|select|sevrm2|critic2>
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; O=outputs/rt/gen_v2_grpo; B=$O/blind_fill_r2
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset VLLM_API_KEY
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
source $S/model_env.sh minicpm5-2b > /dev/null && MC=$MODEL_LOCAL
t() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
CR=rm1__core_r2_bfcl; NEW=rm1__core_rd2
TEMPS="sev=sev__v3c0,rm2v3=rm2__v3b,rm1=$CR"
RMS="sev=$FS/adapters/sev__v3c0-mc/adapter rm2v3=$FS/adapters/rm2__v3b-mc/adapter rm1=$FS/adapters/$CR-mc/adapter"
MECHS="certainty_or_scope_shift=300 near_synonym_shift=300 wrong_attribution=300 merged_across_passages=300"
BF="--round blind_fill_r2 --pools g=rag_pool_r2/prompts.jsonl --verify $MECHS --control"
lora() {   # 名字 源目录 → 本地盘
  [ -f $A/lora/$1/adapter_model.safetensors ] || { mkdir -p $A/lora/$1; local s=$2; [ -d $s/policy ] && s=$s/policy; rsync -a $s/ $A/lora/$1/; }
}
case $1 in
  pool)
    lora rag $FS/adapters/mc-grpo-r3_core_cont4/step100
    free_gpu
    A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-r2a.log $RMS rag=$A/lora/rag || exit 1
    [ -s $O/rag_pool_r2/answers.jsonl ] || python -u -m ragtruth.gen_pool_rag answers --out_dir rag_pool_r2 --model rag --tokenizer $MC --k 8 --temperature 1.0 || exit 1
    python -u -m ragtruth.gen_pool_rag clean --out_dir rag_pool_r2 --tokenizer $MC --thr 0.1 --temps $TEMPS || exit 1
    free_gpu ;;
  samp_v2h3|samp_g1)
    g=${1#samp_}
    [ $g = v2h3 ] && lora v2h3 $FS/adapters/gen-v2h3-mc/adapter || lora g1 $O/g1/ckpt/step50
    free_gpu
    A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-r2$g.log $RMS $g=$A/lora/$g || exit 1
    python -u -m ragtruth.blind_fill sample $BF --model $g --tag $g --tokenizer $MC --temps $TEMPS || exit 1
    free_gpu ;;
  select)
    python -u -m ragtruth.blind_fill select $BF --strat || exit 1 ;;
  sevrm2)
    bash $S/run_sevrm2_r1.sh; [ -f outputs/rt/rm/sev__v3d/eval/report.json ] && [ -f outputs/rt/rm/rm2__v3d/eval/report.json ] || exit 1 ;;
  critic2)
    python -u -m ragtruth.blind_fill build $BF --base $CR --name $NEW || exit 1
    if [ ! -f outputs/rt/rm/$NEW/eval/report.json ]; then free_gpu; RM=$NEW GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh; free_gpu; fi
    [ -f outputs/rt/rm/$NEW/eval/report.json ] || exit 1
    T=$(t $NEW)
    A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-r2c.log c=$FS/adapters/$NEW-mc/adapter || exit 1
    PIDS=""
    for d in outputs/rt/minicpm/grpo300 outputs/rt/minicpm/grpo300b outputs/rt/minicpm/grpo300c; do
      [ -f $d/critic_$NEW.json ] && continue
      python -u -m ragtruth.critic_score --answers $d/answers.jsonl --url http://127.0.0.1:8000/v1 --model c --temp $T --out $d/critic_$NEW.json &
      PIDS="$PIDS $!"
    done
    for p in $PIDS; do wait $p; done
    free_gpu ;;
  *) echo "unknown task $1"; exit 2 ;;
esac
