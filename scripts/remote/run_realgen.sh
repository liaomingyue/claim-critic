#!/bin/bash
# 用真实幻觉句 SFT 错误生成器（ragtruth.realgen），并和 G0 / G2 在同一批种子上对比。
# 在新开的机器上跑（bootstrap_worker.sh 角色 realgen，只拉需要的模型；主实例无卡开机当数据源）。
# 1. 35B 修复真实幻觉句：先修随机 300 句（REPAIR_PARTIAL，本地下载后用 Claude 抽查），接着修完全部（可续跑）；
# 2. 等本地放行（touch outputs/rt/realgen/audit.ok）→ 构建 SFT 样本 → SFT 得 G_sft（ckpt/gen-sft-real）；
# 3. 评测：G0（原始 9B）、G2（gen-dpo-r2）、G_sft 在同一批 300 个新种子上各采样 4 条 → 35B 判有效
#    → D1 打真实性（D1 只见过 35B 合成池，对三个生成器都公平；D2 训练时见过 G0 / G1 的产出，对 G_sft 有偏，不用）
#    → C2 打难度。
# 日志 /root/autodl-tmp/realgen.log，结束写 REALGEN_DONE。
set -u
cd /root/autodl-tmp/critic_model
LOG=/root/autodl-tmp/realgen.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/root/autodl-tmp/models; CK=/root/autodl-tmp/ckpt; O=outputs/rt/realgen; B9=$M/Qwen3.5-9B
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

if ! done_mark repair; then
  serve $M/Qwen3.5-35B-A3B-GPTQ-Int4 judge 16384
  python -m ragtruth.realgen repair --limit 300 --out $O/repair.jsonl >> $LOG 2>&1 || { log "repair FAILED"; exit 1; }
  log "REPAIR_PARTIAL"
  python -m ragtruth.realgen repair --out $O/repair.jsonl >> $LOG 2>&1 || { log "repair FAILED"; exit 1; }
  mark repair
fi
stop
log "waiting for audit.ok"
until [ -f $O/audit.ok ]; do sleep 60; done

if ! done_mark sft; then
  python -m ragtruth.realgen build --repair $O/repair.jsonl --out_dir $O >> $LOG 2>&1 || { log "build FAILED"; exit 1; }
  python -u -m train.sft --config configs/rt/sft_gen_real_9b.yaml --mode coldstart > /root/autodl-tmp/gen-sft-real.log 2>&1 \
    || { log "SFT FAILED"; exit 1; }
  fix_tok $CK/gen-sft-real
  mark sft
fi
if [ ! -f $CK/discrim-9b/model.safetensors ]; then   # D1 只留了适配器，评测前合并回来
  python -u -m train.merge_lora --config configs/rt/discrim_9b.yaml > /root/autodl-tmp/discrim-9b-merge.log 2>&1 \
    || { log "D1 merge FAILED"; exit 1; }
fi

# 评测：同一批种子（seed 99，排除 GAN 两轮用过的种子），每个生成器 4 条，统一 prompt（不带难样本示例）
GENS="g0:$B9 g2:$CK/gen-dpo-r2 gsft:$CK/gen-sft-real"
EX="outputs/rt/gan/r1/samples.jsonl outputs/rt/gan/r2/samples.jsonl"
for gp in $GENS; do
  n=${gp%%:*}; p=${gp#*:}; D=$O/eval_$n
  if ! done_mark eval_$n.sample; then
    fix_tok $p; serve $p gen 16384
    $G sample --dir $D --n_seeds 300 --k 4 --variants 1 --hard_shots 0 --seed 99 --exclude_samples $EX >> $LOG 2>&1 \
      || { log "sample $n FAILED"; exit 1; }
    mark eval_$n.sample
  fi
done
serve $M/Qwen3.5-35B-A3B-GPTQ-Int4 judge 16384
for n in g0 g2 gsft; do
  done_mark eval_$n.valid || { $G valid --dir $O/eval_$n --model judge >> $LOG 2>&1 && mark eval_$n.valid; }
done
fix_tok $CK/discrim-9b; serve $CK/discrim-9b discrim
for n in g0 g2 gsft; do
  done_mark eval_$n.d1 || { $G realism --dir $O/eval_$n --model discrim --tag d1 >> $LOG 2>&1 && mark eval_$n.d1; }
done
fix_tok $CK/critic-gan-r2; serve $CK/critic-gan-r2 critic
for n in g0 g2 gsft; do
  done_mark eval_$n.hard || { $G hard --dir $O/eval_$n --model critic --tag c2 >> $LOG 2>&1 && mark eval_$n.hard; }
done
stop
log "REALGEN_DONE"
