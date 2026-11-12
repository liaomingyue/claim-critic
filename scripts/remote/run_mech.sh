#!/bin/bash
# critic 当奖励的机制分析（ragtruth.reward_mech）：四个起点策略各在固定的 200 道训练题上采 16 条（同训练的采样设定），
# 用历次当过奖励的 critic 逐句打分，再用 9B 金标 critic 打标签。单卡顺序跑：先 2B 服务（采样 + 2B 打分），再 9B 服务（金标 + 9B 奖励 critic）。
#   起点：sft = rag-sft-core；s1 = RAG-S1；s2 = RAG-S2；s3bf = RAG-S3-bf。题目和参考回答用各起点训练时实际用的那份
#   （s3bf 沿用 loop2_prompts_step100：从 RAG-S3-bf 出发的那次训练实际读的就是这一份）。
# 结果：outputs/rt/analysis/mech/{samples_<起点>.jsonl, scores_<起点>_2b.jsonl, scores_<起点>_9b.jsonl, report.json}
set -u
cd ${CODE:-${FS_ROOT:-/root/autodl-fs}/code/critic_model}
A=${TMP_ROOT:-/root/autodl-tmp}; FS=${FS_ROOT:-/root/autodl-fs}; S=scripts/remote; C=outputs/rt/core; LOG=$A/mech.log
L=$A/mech; O=outputs/rt/analysis/mech; K=${K:-200}; N=${N:-16}
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
save() { mkdir -p $O && rsync -a $L/ $O/; }
mkdir -p $L $O $A/s0 $A/s1; rsync -a $O/ $L/   # 接着上次的结果跑（采样和打分都按题断点续跑）
MC=$A/models/MiniCPM5-2B
[ -f $MC/config.json ] || { mkdir -p $A/models && rsync -a $FS/models/MiniCPM5-2B $A/models/; }
source $S/models9b.sh; start_9b_bg
# 用到的适配器和题目先拷到本地盘（共享盘随机读很慢，服务启动和每次读题都会卡）
AD=$A/mech_ad; mkdir -p $AD $A/mech_prompts
for d in rag-sft-core/adapter mc-grpo-r3_core_coreonly/step200/policy mc-grpo-r3_core_cont/step100/policy mc-grpo-r3_core_cont4/step100/policy          rm1__core-mc/adapter rm1__core_r1-mc/adapter rm1__core_r2-mc/adapter rm1__core_r3-mc/adapter rm1__core_r2_bfcl-mc/adapter rm1__core_rd2-mc/adapter          sat_pair__core-mc/adapter refuse__core-mc/adapter rm1__rm1_gold_9b-9b/adapter rm1__core_9b-9b/adapter sat_pair__core_9b-9b/adapter refuse__core_9b-9b/adapter; do
  mkdir -p $AD/$d && rsync -a $FS/adapters/$d/ $AD/$d/ || { log "copy adapter $d FAILED"; exit 1; }
done
for d in rag-sft-core step200 step100; do mkdir -p $A/mech_prompts/$d && rsync -a $C/loop2_prompts_$d/train_ref.jsonl $A/mech_prompts/$d/ || { log "copy prompts $d FAILED"; exit 1; }; done
log "adapters and prompts copied to local disk: $(du -sh $AD | cut -f1)"
declare -A POL=( [sft]=$AD/rag-sft-core/adapter [s1]=$AD/mc-grpo-r3_core_coreonly/step200/policy [s2]=$AD/mc-grpo-r3_core_cont/step100/policy [s3bf]=$AD/mc-grpo-r3_core_cont4/step100/policy )
declare -A PRM=( [sft]=$A/mech_prompts/rag-sft-core [s1]=$A/mech_prompts/step200 [s2]=$A/mech_prompts/step100 [s3bf]=$A/mech_prompts/step100 )
STARTS="${STARTS:-sft s1 s2 s3bf}"
CR2="rm1__core rm1__core_r1 rm1__core_r2 rm1__core_r3 rm1__core_r2_bfcl rm1__core_rd2"

if [ "${SKIP_2B:-0}" != 1 ]; then
  free_gpu
  LORAS=""; CRARG=""
  for s in $STARTS; do LORAS+=" pol_$s=${POL[$s]}"; done
  for c in $CR2; do LORAS+=" $c=$AD/$c-mc/adapter"; CRARG+=" $c=$c"; done
  A=$A/s0 MAX_LORAS=12 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-mech2b.log \
    $LORAS sat=$AD/sat_pair__core-mc/adapter refuse=$AD/refuse__core-mc/adapter >> $LOG 2>&1 || { log "serve 2b FAILED"; exit 1; }
  for s in $STARTS; do
    python -u -m ragtruth.reward_mech sample --model pol_$s --tokenizer $MC --prompts ${PRM[$s]}/train_ref.jsonl --k $K --n $N --out $L/samples_$s.jsonl >> $LOG 2>&1 || { log "sample $s FAILED"; exit 1; }
    log "sampled $s: $(wc -l < $L/samples_$s.jsonl) prompts"
  done
  save
  for s in $STARTS; do
    python -u -m ragtruth.reward_mech score --samples $L/samples_$s.jsonl --prompts ${PRM[$s]}/train_ref.jsonl --critics $CRARG \
      --sat sat=sat_pair__core --refuse refuse=refuse__core --groups 16 --out $L/scores_${s}_2b.jsonl >> $LOG 2>&1 || { log "score 2b $s FAILED"; exit 1; }
    log "scored 2b $s"; save
  done
fi

ensure_9b || { log "9b copy FAILED"; exit 1; }
free_gpu
A=$A/s1 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $A/models/Qwen3.5-9B --log $A/vllm-mech9b.log \
  gold=$AD/rm1__rm1_gold_9b-9b/adapter core9=$AD/rm1__core_9b-9b/adapter sat9=$AD/sat_pair__core_9b-9b/adapter refuse9=$AD/refuse__core_9b-9b/adapter >> $LOG 2>&1 || { log "serve 9b FAILED"; exit 1; }
for s in $STARTS; do
  python -u -m ragtruth.reward_mech score --samples $L/samples_$s.jsonl --prompts ${PRM[$s]}/train_ref.jsonl --critics gold=rm1__rm1_gold_9b core9=rm1__core_9b \
    --sat sat9=sat_pair__core_9b --refuse refuse9=refuse__core_9b --concurrency 48 --groups 6 --out $L/scores_${s}_9b.jsonl >> $LOG 2>&1 || { log "score 9b $s FAILED"; exit 1; }
  log "scored 9b $s"; save
done
free_gpu
python -u -m ragtruth.reward_mech report --dir $L --starts $STARTS --prev s1=rm1__core s2=rm1__core_r1 s3bf=rm1__core_r2_bfcl --out $L/report.json >> $LOG 2>&1 || { log "report FAILED"; exit 1; }
save; cp $LOG $O/
log "MECH_DONE"
