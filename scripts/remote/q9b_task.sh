#!/bin/bash
# 9B 对照实验的任务（队列 scripts/remote/q9b.tasks，queue_worker.sh 按依赖领取）。同一套数据、同一套超参，只把基座从 MiniCPM5-2B 换成 Qwen3.5-9B。
#   prep              9B 版 RAG SFT 数据（prompt 要按 Qwen 的聊天模板重新渲染）；打分模型的数据目录 <名字>_9b（复制 2B 版的数据，评测报告和温度各写各的）
#   rm <名字>         训 9B 打分模型 <名字>_9b（run_rmtrain.sh，超参 configs/rt/rm_9b.yaml），adapter 存 adapters/<名字>_9b-9b
#   sft_rag           9B RAG SFT（configs/rt/sft_policy_9b.yaml）→ adapters/rag-sft-core_9b/adapter
#   gen               9B 错误生成器 v2（configs/rt/sft_gen_v2_9b_same.yaml）→ adapters/gen-v2_9bs-9b；
#                     和 2B 生成器在同样 64 个留出提示 × 32 条上试采，都交给同一套 2B 打分模型（sev / RM2 / critic r2）
#   grpo <组>         RAG GRPO 2×2 的一组（p9r9 / p9r2 / p2r9；p2r2 就是已有的 RAG-S1）：从各自的 SFT 出发、R3 奖励、critic rm1__core（对应大小）、200 步
#   cmp               2×2 四组在三套验收原文上的 9B 评测，配对比较 → results/q9b_2x2_cmp.json
set -u
cd ${CODE:-${FS_ROOT:-/root/autodl-fs}/code/critic_model}
A=${TMP_ROOT:-/root/autodl-tmp}; FS=${FS_ROOT:-/root/autodl-fs}; S=scripts/remote
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
log() { echo "$(date '+%F %T') q9b $*"; }
source $S/models9b.sh
B9=$A/models/Qwen3.5-9B
RMS9="rm1__core sat_pair__core refuse__core"   # 只做 RAG 这一组：critic、满意度、拒答
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }

case $1 in
prep)
  ensure_9b || exit 1
  mkdir -p outputs/rt/core_9b
  python - $B9 <<'EOF' || exit 1
import json, sys
from pathlib import Path
from transformers import AutoTokenizer
from common.schema import read_jsonl
from ragtruth.core import d1_rag_sft
core = list(read_jsonl("outputs/rt/core/core.jsonl"))
print("D1 9B:", d1_rag_sft(core, AutoTokenizer.from_pretrained(sys.argv[1]), Path("outputs/rt/core_9b")))
EOF
  for n in $RMS9; do
    d=outputs/rt/rm/${n}_9b; mkdir -p $d
    for f in train.jsonl calib.jsonl test.jsonl meta.json; do [ -f outputs/rt/rm/$n/$f ] && cp outputs/rt/rm/$n/$f $d/; done
    touch $d/READY
  done
  log "PREP_DONE" ;;
rm)
  n=$2
  [ -f $FS/adapters/${n}_9b-9b/adapter/adapter_model.safetensors ] && [ -f outputs/rt/rm/${n}_9b/eval/report.json ] && { log "rm $n exists"; exit 0; }
  ensure_9b || exit 1; free_gpu
  mkdir -p configs/rt/exp
  RM=${n}_9b bash $S/run_rmtrain.sh || exit 1
  [ -f outputs/rt/rm/${n}_9b/eval/report.json ] || exit 1
  mkdir -p $FS/adapters/${n}_9b-9b && rsync -a $A/ckpt/${n}_9b-9b/adapter $FS/adapters/${n}_9b-9b/ || exit 1
  log "RM_DONE $n" ;;
sft_rag)
  [ -f $FS/adapters/rag-sft-core_9b/adapter/adapter_model.safetensors ] && exit 0
  ensure_9b || exit 1; free_gpu
  mkdir -p configs/rt/exp
  sed -e "s|data: .*|data: outputs/rt/core_9b/rag_sft_train.jsonl|" -e "s|output_dir: .*|output_dir: $A/ckpt/rag-sft-core_9b|" configs/rt/sft_policy_9b.yaml > configs/rt/exp/rag-sft-core_9b.yaml
  python -u -m train.sft --config configs/rt/exp/rag-sft-core_9b.yaml --mode coldstart || exit 1
  mkdir -p $FS/adapters/rag-sft-core_9b && rsync -a $A/ckpt/rag-sft-core_9b/adapter $FS/adapters/rag-sft-core_9b/ || exit 1
  cp $A/ckpt/rag-sft-core_9b/run_manifest.json $A/ckpt/rag-sft-core_9b/sft_meta.json $FS/adapters/rag-sft-core_9b/ 2>/dev/null
  log "SFT_RAG_DONE" ;;
gen)
  ensure_9b || exit 1; free_gpu
  G=$FS/adapters/gen-v2_9bs-9b
  if [ ! -f $G/adapter/adapter_model.safetensors ]; then
    mkdir -p configs/rt/exp
    sed -e "s|output_dir: .*|output_dir: $A/ckpt/gen-v2_9bs|" configs/rt/sft_gen_v2_9b_same.yaml > configs/rt/exp/gen-v2_9bs.yaml
    python -u -m train.sft --config configs/rt/exp/gen-v2_9bs.yaml --mode coldstart || exit 1
    mkdir -p $G && rsync -a $A/ckpt/gen-v2_9bs/adapter $G/ || exit 1
  fi
  [ -f $A/models/MiniCPM5-2B/config.json ] || { mkdir -p $A/models && rsync -a $FS/models/MiniCPM5-2B $A/models/; }
  MC=$A/models/MiniCPM5-2B
  SC="sev=$FS/adapters/sev__v3c0-mc/adapter rm2v3=$FS/adapters/rm2__v3b-mc/adapter rm1=$FS/adapters/rm1__core_r2-mc/adapter"
  G1=$(nvidia-smi -L | wc -l); [ $G1 -ge 2 ] && G1=1 || G1=0   # 双卡机器上 9B 生成器放 GPU1
  # 打分模型（2B）一个服务；9B 生成器一个服务；2B 生成器（gen-v2-mc）挂在 2B 打分服务上，同一套提示和打分
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8001 --mem 0.3 --base $MC --log $A/vllm-q9b-sc.log $SC gen2=$FS/adapters/gen-v2-mc/adapter || exit 1
  A=$A/s1 bash $S/serve_loras.sh --gpu $G1 --port 8000 --mem 0.55 --base $B9 --log $A/vllm-q9b-gen.log gen9=$G/adapter || exit 1
  python -u -m ragtruth.gen_v2_grpo probe --model gen9 --url http://127.0.0.1:8000/v1 --rm_url http://127.0.0.1:8001/v1 --tokenizer $B9 || exit 1
  python -u -m ragtruth.gen_v2_grpo probe --model gen2 --url http://127.0.0.1:8001/v1 --tokenizer $MC || exit 1
  free_gpu
  log "GEN_DONE" ;;
grpo)
  arm=$2
  case $arm in
    p9r9) PMx=qwen3.5-9b; RMx=qwen3.5-9b; INITx=$FS/adapters/rag-sft-core_9b/adapter; SFX=_9b ;;
    p9r2) PMx=qwen3.5-9b; RMx=minicpm5-2b; INITx=$FS/adapters/rag-sft-core_9b/adapter; SFX= ;;
    p2r9) PMx=minicpm5-2b; RMx=qwen3.5-9b; INITx=$FS/adapters/rag-sft-core/adapter; SFX=_9b ;;
    *) echo "bad arm $arm"; exit 1 ;;
  esac
  SM=32; [ $PMx = qwen3.5-9b ] && SM=8   # 9B 训练每批少接几条 completion，防显存不够
  POLICY_MODEL=$PMx RM_MODEL=$RMx ARM=q9b_$arm CRITIC=rm1__core$SFX SAT=sat_pair__core$SFX REFUSE=refuse__core$SFX INIT=$INITx \
    STEPS=200 EVAL_STEPS="100 200" EVAL_V0=1 SHARE_MB=$SM CODE=$PWD bash $S/run_loop2.sh
  [ -f outputs/rt/core/loop2_q9b_$arm/eval/s200_c_9b.json ] || exit 1
  log "GRPO_DONE $arm" ;;
ref9)   # 双机模式要提前生成参考回答（9B SFT 起点在训练原文上重采，定长度范围和满意度参考），GRPO 时直接用
  POLICY_MODEL=qwen3.5-9b ARM=q9b_p9r9 CRITIC=rm1__core_9b INIT=$FS/adapters/rag-sft-core_9b/adapter REF_ONLY=1 CODE=$PWD bash $S/run_loop2.sh || exit 1
  log "REF9_DONE" ;;
pair_infer)   # 两台单卡跑 9B GRPO 的推理机：9B 底座 + 9B critic / 满意度 / 拒答 + 策略起点；等训练机写 train_done 才退出
  P=$FS/runs/pair/q9b; mkdir -p $P
  rm -f $P/infer_ready $P/infer_failed $P/train_done $P/train_failed $P/domains.json
  hostname > $P/infer_host
  printf '%s\n' "critic=$FS/adapters/rm1__core_9b-9b/adapter" "sat=$FS/adapters/sat_pair__core_9b-9b/adapter" \
    "refuse=$FS/adapters/refuse__core_9b-9b/adapter" "policy_v0=$FS/adapters/rag-sft-core_9b/adapter" > $P/loras
  PAIR=q9b INFER_MODEL=qwen3.5-9b bash $S/run_infer_node.sh || exit 1
  [ -f $P/train_done ] || exit 1
  log "PAIR_INFER_DONE" ;;
pair_train)   # 两台单卡跑 9B GRPO 的训练机：等推理机就绪（域名由本地 scripts/pair_setup.py 写），跑 9B × 9B 的 GRPO，训完在本机评测
  PAIR=q9b JOB=run_q9b_pair.sh bash $S/run_train_node.sh || exit 1
  log "PAIR_TRAIN_DONE" ;;
cap3)   # 9B critic 和 2B critic 在 Claude 金标验收集上直接比（同 critic_cap3 口径）
  ensure_9b || exit 1; free_gpu
  T=$(python -c "import json;print(json.load(open('outputs/rt/rm/rm1__core_9b/eval/report.json'))['temperature'])")
  J="["; for s in grpo300 grpo300b grpo300c; do J+="{\"model\":\"c9\",\"temp\":$T,\"answers\":\"outputs/rt/minicpm/$s/answers.jsonl\",\"out\":\"outputs/rt/minicpm/$s/critic_rm1__core_9b.json\"},"; done
  echo "${J%,}]" > $A/q9b_cap3_jobs.json
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $B9 --log $A/vllm-q9b-cap3.log c9=$FS/adapters/rm1__core_9b-9b/adapter || exit 1
  python -u -m ragtruth.critic_score_multi --url http://127.0.0.1:8000/v1 --jobs $A/q9b_cap3_jobs.json --concurrency 256 || exit 1
  free_gpu
  python -m ragtruth.critic_cap3 --critics rm1__core rm1__core_9b --base rm1__core --by_type --out results/q9b_critic_cap3.json || exit 1
  log "CAP3_DONE" ;;
cmp)   # 9B 全套 vs 2B 全套（2B 全套 = 已有的 RAG-S1：从 RAG-SFT 出发、critic rm1__core、200 步）
  E=outputs/rt/core
  python -m ragtruth.pol9b_cmp --arm sft2=$E/loop2_coreonly/eval:v0 --arm sft9=$E/loop2_q9b_p9r9/eval:v0 \
    --arm s1_2b=$E/loop2_coreonly/eval:s200 --arm s100_9b=$E/loop2_q9b_p9r9/eval:s100 --arm s200_9b=$E/loop2_q9b_p9r9/eval:s200 \
    --base s1_2b --n_boot 1000 --out results/q9b_cmp.json || exit 1
  log "CMP_DONE" ;;
*) echo "usage: q9b_task.sh prep|rm <名字>|sft_rag|cap3|grpo <组>|cmp"; exit 1 ;;
esac
