#!/bin/bash
# 边界样本 critic 当奖励，重跑 2B 回答 GRPO（看边界样本提升 critic 以后，RAG 生成能不能比上一版提升更大；critic 只用 h2；先不用 Claude）。
# 和上一版（outputs/rt/minicpm/grpo，rm1__mcgold，第 300 步）唯一的区别是 critic：
#   进奖励：rm1__mcgold_h2（幻觉）、sat_pair__mc（满意度护栏，只罚，w 0.5）、refuse（拒答 RM，w 0.4）、硬规则
#   只监控：9B 金标 critic rm1__rm1_gold_9b（训完补打 dev 评测样本 + 验收回答）
# 同一批 prompt / 长度参考 / 满意度参考（复制上一版的文件），lr 5e-6，300 步。
# 训完：① 9B 补打 dev 评测样本（和上一版 shadow9b 按原文配对）；② 新旧两版第 300 步在 grpo300 / grpo300b 的 200 个原文-种子上重新生成，9B critic 逐句打分。机器不关。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/grpoh2.log; S=$(dirname $(readlink -f $0)); MC=$FS/models/MiniCPM5-2B
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OLD=outputs/rt/minicpm/grpo; O=outputs/rt/minicpm/grpo_h2; C=outputs/rt/minicpm/cmp_h2
mkdir -p $O $C $A/s0
for f in train_all.jsonl dev.jsonl train.jsonl train.meta.json train_ref.jsonl dev_ref.jsonl; do
  [ -f $OLD/$f ] || { log "missing $OLD/$f"; exit 1; }
  cp $OLD/$f $O/
done
for f in $OLD/*.json*; do [ -f $O/$(basename $f) ] || cp $f $O/; done
# 检查点是 save_pretrained(selected_adapters=["policy"]) 存的，LoRA 在 step300/policy/ 下（踩过：只看 step300/ 误判缺失）
ad() { [ -f $1/policy/adapter_model.safetensors ] && echo $1/policy || echo $1; }
OLDAD=$(ad $FS/adapters/mc-grpo/step300)
[ -f $OLDAD/adapter_model.safetensors ] || { log "missing old step300 adapter ($OLDAD)"; exit 1; }
[ -f outputs/rt/rm/rm1__mcgold_h2/eval/report.json ] || { log "missing h2 report"; exit 1; }
log "start: critic rm1__mcgold_h2, prompts / refs copied from $OLD"
CRITIC=rm1__mcgold_h2 LR=5e-6 STEPS=300 O=$O ADOUT=mc-grpo-h2 bash $S/run_mcgrpo.sh || { log "grpo FAILED"; exit 1; }
grep -aq "train done" $A/mcgrpo.log || { log "grpo train FAILED (see mcgrpo.log)"; exit 1; }
log "grpo done"
# 双机模式（run_train_node.sh）：训完后的补打和验收都在训练机本地卡上跑，不再用推理机，本地服务不要密钥
if [ -n "${RM_URL:-}" ]; then unset RM_URL LORA_SYNC_URL VLLM_API_KEY PUSH_DIR; export OPENAI_API_KEY=x; fi
G=$([ "${SINGLE_GPU:-0}" = 1 ] && echo 0 || echo 1)
RUN=$O/run DEV=$O/dev_ref.jsonl bash $S/run_shadow9b.sh >> $LOG 2>&1; log "shadow9b done: $(tail -1 $O/run/shadow9b.jsonl 2>/dev/null | cut -c1-400)"
# 验收原文上新旧两版重新生成（同 system / 种子；grpo300 T 1.0、grpo300b T 0.7，top_p 都是 0.95 = MiniCPM generation_config）
pkill -f '[v]llm serve'; sleep 10
A=$A bash $S/serve_loras.sh --gpu $G --port 8000 --base $MC --log $A/vllm-acc.log old=$OLDAD new=$(ad $FS/adapters/mc-grpo-h2/step300) >> $LOG 2>&1 || { log "serve policies FAILED"; exit 1; }
for m in old new; do
  python -u -m ragtruth.mc_accept_gen --ref outputs/rt/minicpm/grpo300/answers.jsonl --temperature 1.0 --top_p 0.95 --url http://127.0.0.1:8000/v1 --model $m --out $C/${m}_a.jsonl >> $LOG 2>&1 || log "gen $m a FAILED"
  python -u -m ragtruth.mc_accept_gen --ref outputs/rt/minicpm/grpo300b/answers.jsonl --temperature 0.7 --top_p 0.95 --url http://127.0.0.1:8000/v1 --model $m --out $C/${m}_b.jsonl >> $LOG 2>&1 || log "gen $m b FAILED"
done
pkill -f '[v]llm serve'; sleep 10
CR=rm1__rm1_gold_9b; T=$(python -c "import json;print(json.load(open('outputs/rt/rm/$CR/eval/report.json'))['temperature'])")
A=$A bash $S/serve_loras.sh --gpu $G --port 8000 --base $FS/models/Qwen3.5-9B --log $A/vllm-acc9.log critic=$FS/adapters/$CR-9b/adapter >> $LOG 2>&1 || { log "serve 9b FAILED"; exit 1; }
for f in old_a old_b new_a new_b; do
  python -u -m ragtruth.critic_score --answers $C/$f.jsonl --url http://127.0.0.1:8000/v1 --temp $T --out $C/${f}_9b.json >> $LOG 2>&1 || log "score $f FAILED"
done
pkill -f '[v]llm serve'
log "GRPOH2_DONE"
