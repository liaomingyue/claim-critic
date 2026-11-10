#!/bin/bash
# R1：用发布的 critic LoRA 在三套验收回答上重新逐句打分，和发布的打分 / 结论对比（1 张卡，约半小时）。
#   bash scripts/repro/r1_rescore.sh
# 新分数写到 outputs/rt/minicpm/grpo300{,b,c}/critic_<名字>_re.json（不覆盖发布的 critic_<名字>.json）。
# vLLM 的 logprob 有很小的数值抖动，所以逐句分数不会逐位相同；compare_r1.py 看三件事：逐句分数是否几乎一样、AUC 差多少、关键结论是否成立。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"; cd "$ROOT"
export OPENAI_API_KEY=x
CRITICS=${CRITICS:-"rm1__core rm1__core_r1 rm1__core_r2 rm1__core_r2_bfcl rm1__core_r2_bfnc rm1__core_rd2"}
source scripts/remote/model_env.sh minicpm5-2b || exit 1
L=""; J="["
for c in $CRITICS; do
  L+=" $c=$FS_ROOT/adapters/$c-mc/adapter"
  T=$(python -c "import json;print(json.load(open('outputs/rt/rm/$c/eval/report.json'))['temperature'])")
  for d in grpo300 grpo300b grpo300c; do
    J+="{\"model\":\"$c\",\"temp\":$T,\"answers\":\"outputs/rt/minicpm/$d/answers.jsonl\",\"out\":\"outputs/rt/minicpm/$d/critic_${c}_re.json\"},"
  done
done
echo "${J%,}]" > $TMP_ROOT/r1_jobs.json
rlog "R1: serve $(echo $CRITICS | wc -w) critics on $MODEL_LOCAL"
MAX_LORAS=8 bash scripts/remote/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MODEL_LOCAL --log $TMP_ROOT/vllm-r1.log $L || exit 1
python -u -m ragtruth.critic_score_multi --url http://127.0.0.1:8000/v1 --jobs $TMP_ROOT/r1_jobs.json --concurrency 256 > $TMP_ROOT/r1_score.log 2>&1 || { rlog "score FAILED"; tail -5 $TMP_ROOT/r1_score.log; exit 1; }
pkill -f '[v]llm serve'
ALL=""; for c in $CRITICS; do ALL+=" $c ${c}_re"; done
python -m ragtruth.critic_cap3 --critics $ALL --base rm1__core_r2_bfcl_re --by_type --out $TMP_ROOT/r1_cap3.json > $TMP_ROOT/r1_cap3.txt 2>&1 || { rlog "cap3 FAILED"; exit 1; }
python -m scripts.repro.compare_r1 --cap3 $TMP_ROOT/r1_cap3.json --critics $CRITICS | tee $TMP_ROOT/r1_report.txt
rlog "R1_DONE（报告：$TMP_ROOT/r1_report.txt）"
