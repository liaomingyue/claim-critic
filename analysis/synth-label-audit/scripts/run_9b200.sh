#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-fs/code/critic_model || exit 1
export OPENAI_API_KEY=x PYTHONHASHSEED=0
M9=/root/autodl-fs/models/Qwen3.5-9B
# 重抽同一批200(seed42),带原始 context 列表
python - <<'PY'
import json, collections, random
H="outputs/rt/gentarget/harvest_gated.jsonl"
xs=[x for x in (json.loads(l) for l in open(H)) if x.get("p_halluc") is not None and x["p_correct"]>=0.5 and x["orig_p"]<=0.5 and 0.0<=x["p_halluc"]<0.5]
xs.sort(key=lambda x:x["id"]); rng=random.Random(0); rng.shuffle(xs)
seen=set(); per=collections.Counter(); keep=[]
for x in xs:
    if x["sentence"] in seen or per[x["prompt_id"]]>=2: continue
    seen.add(x["sentence"]); per[x["prompt_id"]]+=1; keep.append(x)
    if len(keep)>=2000: break
samp=random.Random(42).sample(keep,200)
json.dump([{"id":x["id"],"source":x["source"],"idx":x["idx"],"context":x["context"]} for x in samp], open("/root/eval200b.json","w"))
print("re-extracted", len(samp))
PY
pkill -9 -f vllm 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null; sleep 5
for i in $(seq 1 40); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "$m" -lt 1500 ] && break; sleep 3; done
echo "$(date '+%F %T') serve 9B"
vllm serve $M9 --served-model-name critic --max-model-len 16384 --gpu-memory-utilization 0.85 --port 8000 --max-logprobs 20 > /root/autodl-tmp/vllm-9b.log 2>&1 &
V=$!
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q critic && break; kill -0 $V 2>/dev/null || { echo SERVE_FAIL; exit 1; }; sleep 10; done
echo "$(date '+%F %T') 9B up, scoring 200"
python - <<'PY'
import json, math
from openai import OpenAI
from ragtruth.task import render_input
from ragtruth.rm_data import RM1_SYSTEM
cl=OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="x")
def p_err(source, sents, idx, T=1.3):
    r=cl.chat.completions.create(model="critic", messages=[{"role":"system","content":RM1_SYSTEM},
        {"role":"user","content":render_input({"source":source,"sentences":sents,"idx":idx})}],
        max_tokens=1, temperature=0, logprobs=True, top_logprobs=20)
    lp={t.token.strip():t.logprob for t in r.choices[0].logprobs.content[0].top_logprobs}
    d={k:lp.get(k,-100)/T for k in ("A","B","C")}; m=max(d.values()); e={k:math.exp(v-m) for k,v in d.items()}; z=sum(e.values())
    return (e["B"]+e["C"])/z
data=json.load(open("/root/eval200b.json")); out=[]
from concurrent.futures import ThreadPoolExecutor
def go(x):
    try: return {"id":x["id"],"p_err_9b":round(p_err(x["source"],x["context"],x["idx"]),4)}
    except Exception as e: return {"id":x["id"],"p_err_9b":None,"err":str(e)[:60]}
with ThreadPoolExecutor(16) as ex: out=list(ex.map(go,data))
json.dump(out, open("/root/p9b_200.json","w"))
print("scored", sum(o["p_err_9b"] is not None for o in out))
PY
pkill -9 -f vllm 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
echo "$(date '+%F %T') 9B200_DONE"
