#!/bin/bash
# 第二轮里其他 5 种类型也适度少量补一些（每类 60）。r2_task.sh 正在被执行（不能改），新增 / 改动的任务放这里：
#   r2_task_b.sh <sampx_v2h3|sampx_g1|select|critic2>
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; O=outputs/rt/gen_v2_grpo
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset VLLM_API_KEY
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
source $S/model_env.sh minicpm5-2b > /dev/null && MC=$MODEL_LOCAL
t() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
CR=rm1__core_r2_bfcl; NEW=rm1__core_rd2
TEMPS="sev=sev__v3c0,rm2v3=rm2__v3b,rm1=$CR"
RMS="sev=$FS/adapters/sev__v3c0-mc/adapter rm2v3=$FS/adapters/rm2__v3b-mc/adapter rm1=$FS/adapters/$CR-mc/adapter"
M4="certainty_or_scope_shift=300 near_synonym_shift=300 wrong_attribution=300 merged_across_passages=300"
M5="contradicts_source=60 unsupported_inference=60 number_or_date=60 vague_evaluation_or_addon=60 added_specific_fact=60"
POOL="--round blind_fill_r2 --pools g=rag_pool_r2/prompts.jsonl"
lora() { [ -f $A/lora/$1/adapter_model.safetensors ] || { mkdir -p $A/lora/$1; local s=$2; [ -d $s/policy ] && s=$s/policy; rsync -a $s/ $A/lora/$1/; }; }
case $1 in
  sampx_v2h3|sampx_g1)
    g=${1#sampx_}
    [ $g = v2h3 ] && lora v2h3 $FS/adapters/gen-v2h3-mc/adapter || lora g1 $O/g1/ckpt/step50
    free_gpu
    A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-r2x$g.log $RMS $g=$A/lora/$g || exit 1
    python -u -m ragtruth.blind_fill sample $POOL --verify $M5 --control --model $g --tag ${g}x --tokenizer $MC --temps $TEMPS || exit 1
    free_gpu ;;
  select)
    python -u -m ragtruth.blind_fill select $POOL --verify $M4 $M5 --control --strat || exit 1 ;;
  critic2)
    python -u -m ragtruth.blind_fill build $POOL --verify $M4 $M5 --control --base $CR --name $NEW || exit 1
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
