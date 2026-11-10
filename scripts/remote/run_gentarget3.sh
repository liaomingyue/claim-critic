#!/bin/bash
# 定向造错 GRPO 第三版（改对要结合整个回答、上下句尤其重要；弄完直接双卡）。相对第二版：
#   种子：172 条，按整条回答和上下句重新改对（上下句有错一起改、不删）；生成器 prompt 加「结合整条回答和上下句」；
#   提示池一半指定 baseless、一半指定 conflict（第二版第 75 步 conflict 占 95% 停训）；造错类型和 prompt 不符记 wrong_type、不给分；
#   RM2（rm2v2__mcgen3）自然一侧补上 Claude 确认的 2B 真错（第二版 RM2 上「错得明显」本身就是区分合成的信号）。
#   SFT 单卡（数据只有 172 条，双卡没收益）；synth 双卡各起一个服务分摊。
# 进奖励：rm2（真实性，主力）、rm3（正确性，门槛 + 扣分）、rm1（2B critic，判出就扣分）；原句有没有错用预先算好的 9B 分数。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/gentarget3.log; S=$(dirname $(readlink -f $0))
MC=$FS/models/MiniCPM5-2B; O=outputs/rt/gentarget; CK=$A/ckpt; INIT=$FS/adapters/gen-seed3-mc/adapter; RM2=rm2v2__mcgen3
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
fail() { log "$1 FAILED"; pkill -f '[v]llm serve'; log "GENTARGET3_DONE"; exit 1; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
temp() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
rep() { log "$1 report $(python -c "import json;r=json.load(open('outputs/rt/rm/$1/eval/report.json'));print(json.dumps({'T':r['temperature'],'test':{k:r['test_calibrated'].get(k) for k in ('n','auc','ece','recall@0.5','precision@0.5')}}))")"; }
mkdir -p $A/s0 $A/s1

log "phase 1"
if [ ! -f $INIT/adapter_model.safetensors ]; then
  ( log "train gen-seed3-mc"
    CUDA_VISIBLE_DEVICES=0 python -u -m train.sft --config configs/rt/sft_gen_seed3_mc.yaml --mode coldstart > $A/gen-seed3-mc.log 2>&1 \
      && mkdir -p $FS/adapters/gen-seed3-mc && rsync -a $CK/gen-seed3-mc/adapter $FS/adapters/gen-seed3-mc/ \
      && log "trained gen-seed3-mc $(grep -o "'train_loss': '[0-9.]*'" $A/gen-seed3-mc.log | tail -1)" ) & P0=$!
else P0=""; fi
[ -f $O/prompts_v2.jsonl ] || { cp $O/prompts.jsonl $O/prompts_v2.jsonl; cp $O/eval_prompts.jsonl $O/eval_prompts_v2.jsonl; cp $O/meta.json $O/meta_v2.json; }
python -u -m ragtruth.gen_target_data build --free_size 1 --ctx 1 --type_split 1 >> $LOG 2>&1 || fail build
[ -n "$P0" ] && wait $P0
[ -f $INIT/adapter_model.safetensors ] || fail "gen-seed3-mc"

log "phase 2"
if [ ! -f outputs/rt/rm/$RM2/eval/report.json ]; then
  if [ ! -f $O/synth_seed3.jsonl ]; then
    # 双卡各起一个同样的服务（评测时两张卡一起用），请求按样本 id 分摊。先等两张卡显存放干净（踩过：上一轮的 vLLM 还在退出，起不来）
    for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done
    for g in 0 1; do
      A=$A/s$g bash $S/serve_loras.sh --gpu $g --port 800$((g+1)) --base $MC --log $A/vllm-gt3-syn$g.log \
        gen=$INIT rm3=$FS/adapters/rm3__mc-mc/adapter rm2old=$FS/adapters/rm2v2__mcgen2-mc/adapter >> $LOG 2>&1 & eval "SV$g=\$!"
    done
    wait $SV0 || fail "synth vllm gpu0"; wait $SV1 || fail "synth vllm gpu1"
    U=http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1
    python -u -m ragtruth.gen_target_data synth --url $U --model gen --rm3 rm3 --rm3_temp $(temp rm3__mc) --samples 2 --out synth_seed3.jsonl >> $LOG 2>&1 || fail synth
    python -u -m ragtruth.gen_target_data rm2score --url $U --model rm2old --rm2_temp $(temp rm2v2__mcgen2) --rm2_name rm2v2__mcgen2 --inp synth_seed3.jsonl >> $LOG 2>&1 || log "rm2score FAILED (continue)"
    pkill -f '[v]llm serve'; sleep 10
  fi
  python -u -m ragtruth.gen_target_data rm2data --inp synth_seed3.jsonl --name $RM2 --seed_errors 1 >> $LOG 2>&1 || fail rm2data
  RM=$RM2 GPU=0 PORT=8100 bash $S/run_mcrm.sh
  [ -f outputs/rt/rm/$RM2/eval/report.json ] || fail $RM2
fi
rep $RM2

log "phase 3"
pkill -f '[v]llm serve'
for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done
A=$A bash $S/serve_loras.sh --gpu 1 --port 8000 --base $MC --log $A/vllm-gt3.log --priority \
  rm1=$FS/adapters/rm1__mcgold-mc/adapter rm2=$FS/adapters/$RM2-mc/adapter rm3=$FS/adapters/rm3__mc-mc/adapter policy_v0=$INIT >> $LOG 2>&1 || fail vllm
log "rm: rm1=rm1__mcgold（2B critic，扣分） rm2=$RM2（真实性，主力） rm3=rm3__mc（正确性，门槛 + 扣分）；原句 orig_p 用 9B 金标（预先算好）"
CUDA_VISIBLE_DEVICES=0 python -u -m train.gen_grpo --causal 1 --base $MC --init $INIT --prompts $O/prompts.jsonl --eval_prompts $O/eval_prompts.jsonl \
  --reward target --temps rm1=rm1__mcgold,rm2=$RM2,rm3=rm3__mc --div_mode place --w_div ${W_DIV:-0.2} --w_corr ${W_CORR:-0.5} --w_crit ${W_CRIT:-0.3} --gate ${GATE:-0.7} \
  --conflict_range 0.25 0.8 --other_max 1.0 --top_max 0.75 --share 1 --share_impl train.prefix_share_llama --share_mb 8 --ref_on_vllm 1 --version_prefix t3_ \
  --steps ${STEPS:-300} --lr ${LR:-5e-6} --n ${N:-8} --prompts_per_step 16 --eval_k 4 --save_every 50 --out $O/grpo3 ${RESUME:+--resume $RESUME} >> $A/gentarget3_grpo.log 2>&1 \
  && log "gen grpo done" || log "gen grpo FAILED"
pkill -f '[v]llm serve'
cp $A/gentarget3_grpo.log $O/grpo3/ 2>/dev/null
log "GENTARGET3_DONE"
