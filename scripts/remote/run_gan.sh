#!/bin/bash
# 生成器—判别器—critic 两轮循环（类似 GAN，用 DPO 代替不能回传的梯度），主实例上顺序执行。
# 交付：两轮后的生成器 G2（gen-dpo-r2）、判别器 D2（discrim-9b-r2）、critic C2（critic-gan-r2），以及中间的 G1 / C1。
# 每轮：G 采样（600 种子 × 8）→ 35B 判有效 → 当前 critic 打难度 → 当前判别器打真实性 → 帕累托偏好对（统一 prompt）
#       → DPO 得新 G；本轮「难 + 真实」前 600 条加入 critic 训练集（累积）→ 从基座训新 critic 并评测；
#       第 2 轮开始前用 G1 的新产出重训判别器（判别器追上生成器）。
# 日志 /root/autodl-tmp/gan.log，结束写 GAN_DONE。
set -u
cd /root/autodl-tmp/critic_model
LOG=/root/autodl-tmp/gan.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/root/autodl-tmp/models; CK=/root/autodl-tmp/ckpt; O=outputs/rt/gan; B9=$M/Qwen3.5-9B
MM='{"image":0,"video":0}'
G="python -m ragtruth.gen_dpo"
mkdir -p $O
stop() { pkill -f '[v]llm serve'; sleep 10; pkill -9 -f '[V]LLM::EngineCore'; sleep 3; }
fix_tok() { cp $B9/{tokenizer_config.json,tokenizer.json,vocab.json,merges.txt,chat_template.jinja,preprocessor_config.json,video_preprocessor_config.json} "$1"/ 2>/dev/null; }
serve() {  # serve <路径> <名字> <最大长度>
  stop
  nohup vllm serve "$1" --served-model-name "$2" --max-model-len ${3:-8192} --gpu-memory-utilization 0.9 --port 8000 \
    --limit-mm-per-prompt "$MM" > /root/autodl-tmp/vllm-$2.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q "$2" && return 0; sleep 10; done
  log "vllm failed for $2"; exit 1
}
done_mark() { [ -f $O/$1.done ]; }
mark() { touch $O/$1.done; log "$1 done"; }

until tail -1 /root/autodl-tmp/discrim.log 2>/dev/null | grep -q DISCRIM_DONE; do sleep 60; done
log "discriminator D1: $(grep -m1 accuracy outputs/rt/discrim/heldout_scores_summary.json 2>/dev/null)"

GEN=$B9; CRITIC=$CK/sft-9b-v2-notips; DISC=$CK/discrim-9b; SELS=""; PREV=""
for r in 1 2; do
  D=$O/r$r
  HS=$([ $r = 1 ] && echo 2 || echo 0)   # 第 1 轮采样带难样本示例；G1 以后用统一 prompt（不带）训练和采样
  if ! done_mark r$r.sample; then
    log "round $r: sample with $(basename $GEN)"; fix_tok $GEN; serve $GEN gen 16384
    $G sample --dir $D --n_seeds 600 --k 8 --variants 4 --hard_shots $HS --seed $((10 + r)) \
      --exclude_samples $PREV >> $LOG 2>&1 || { log "sample FAILED"; exit 1; }
    mark r$r.sample
  fi
  if ! done_mark r$r.valid; then
    serve $M/Qwen3.5-35B-A3B-GPTQ-Int4 judge 16384
    $G valid --dir $D --model judge >> $LOG 2>&1 || { log "valid FAILED"; exit 1; }
    mark r$r.valid
  fi
  if ! done_mark r$r.hard; then
    fix_tok $CRITIC; serve $CRITIC critic
    $G hard --dir $D --model critic --tag target >> $LOG 2>&1 || { log "hard FAILED"; exit 1; }
    mark r$r.hard
  fi
  if [ $r = 2 ] && ! done_mark d2; then   # 判别器追上生成器：G0 / G1 的有效产出都并入合成类，重训
    python -m ragtruth.discrim build --out_dir outputs/rt/discrim_r2 --gen_dirs $O/r1 $O/r2 --cap 4000 >> $LOG 2>&1
    sed -e "s|data: .*|data: outputs/rt/discrim_r2/train.jsonl|" -e "s|output_dir: .*|output_dir: $CK/discrim-9b-r2|" \
      configs/rt/discrim_9b.yaml > configs/rt/exp/discrim_9b_r2.yaml
    stop
    if [ -f $CK/discrim-9b-r2/adapter/adapter_model.safetensors ]; then   # 训完但合并保存失败过：直接从适配器合并，不重训
      python -u -m train.merge_lora --config configs/rt/exp/discrim_9b_r2.yaml --log /root/autodl-tmp/discrim-9b-r2.log \
        > /root/autodl-tmp/discrim-9b-r2-merge.log 2>&1 || { log "D2 merge FAILED"; exit 1; }
    else
      python -u -m train.sft --config configs/rt/exp/discrim_9b_r2.yaml --mode coldstart > /root/autodl-tmp/discrim-9b-r2.log 2>&1 \
        || { log "D2 train FAILED"; exit 1; }
    fi
    fix_tok $CK/discrim-9b-r2; serve $CK/discrim-9b-r2 discrim
    python -m ragtruth.discrim score --rows outputs/rt/discrim_r2/heldout.jsonl --out outputs/rt/discrim_r2/heldout_scores.jsonl \
      --eval --logprob >> $LOG 2>&1
    mark d2
  fi
  [ $r = 2 ] && DISC=$CK/discrim-9b-r2
  if ! done_mark r$r.realism; then
    fix_tok $DISC; serve $DISC discrim
    $G realism --dir $D --model discrim --tag discrim >> $LOG 2>&1 || { log "realism FAILED"; exit 1; }
    mark r$r.realism
  fi
  stop
  $G pairs --dir $D --mode hard_real --canonical >> $LOG 2>&1
  $G select --dir $D --n 600 --out $D/critic_sel.jsonl >> $LOG 2>&1
  $G report --dir $D --tag target >> $LOG 2>&1
  SELS="$SELS $D/critic_sel.jsonl"
  # 生成器 DPO：从上一轮的生成器继续（参考模型 = 本轮起点）
  if ! done_mark r$r.dpo; then
    sed -e "s|^base_model: .*|base_model: $GEN|" -e "s|^data: .*|data: $D/pairs_hard_real_canon.jsonl|" \
      -e "s|^output_dir: .*|output_dir: $CK/gen-dpo-r$r|" configs/rt/dpo_gen_9b.yaml > configs/rt/exp/dpo_gen_r$r.yaml
    log "round $r: DPO generator"
    python -u -m train.dpo --config configs/rt/exp/dpo_gen_r$r.yaml > /root/autodl-tmp/dpo-gen-r$r.log 2>&1 \
      || { log "DPO FAILED"; exit 1; }
    mark r$r.dpo
  fi
  GEN=$CK/gen-dpo-r$r
  # critic：基础数据 + 各轮「难 + 真实」样本（成对、累积），每轮从基座重训，在真实测试集上评测。
  # PARALLEL=1 时交给另一台机器（run_gan_critic.sh），这里只放一个「本轮选样就绪」标记，与生成器 DPO 并行
  name=critic-gan-r$r
  if [ -n "${PARALLEL:-}" ]; then
    touch $D/critic_sel.ready
  elif ! done_mark r$r.critic; then
    python -m ragtruth.iterate build --base outputs/rt/claims/train_sub35_r3.jsonl --sel $SELS \
      --out outputs/rt/claims/gan_r$r.jsonl >> $LOG 2>&1
    sed -e "s|data: .*|data: outputs/rt/claims/gan_r$r.jsonl|" -e "s|output_dir: .*|output_dir: $CK/$name|" \
      configs/rt/sft_9b_v2.yaml > configs/rt/exp/$name.yaml
    log "round $r: train $name"
    python -u -m train.sft --config configs/rt/exp/$name.yaml --mode coldstart > /root/autodl-tmp/$name.log 2>&1 \
      || { log "critic train FAILED"; exit 1; }
    fix_tok $CK/$name; serve $CK/$name $name
    python -m ragtruth.eval --model $name --base_url http://127.0.0.1:8000/v1 --concurrency 128 --no_think \
      --style full --max_tokens 512 --out outputs/rt/eval/sft-9b-$name > /root/autodl-tmp/eval-$name.log 2>&1
    stop
    mark r$r.critic
  fi
  CRITIC=$CK/$name
  PREV="$PREV $D/samples.jsonl"
  if [ -n "${PARALLEL:-}" ]; then   # 下一轮打难度要用本轮的 critic：等另一台机器训完并把它推回来
    log "round $r: waiting for $name from the critic machine"
    until [ -f $CK/$name/.synced ]; do sleep 60; done
    log "round $r: got $name"
  fi
done
log "GAN_DONE"
