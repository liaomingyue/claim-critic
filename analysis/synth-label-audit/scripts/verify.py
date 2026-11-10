import json, subprocess, re
from concurrent.futures import ThreadPoolExecutor
V={json.loads(l)["id"]:json.loads(l) for l in open("/tmp/verdicts.jsonl")}
E={json.loads(l)["id"]:json.loads(l) for l in open("/tmp/ev200.jsonl")}
# 只审 Claude(纯检测)判 faithful 的
sub=[V[i] for i in V if V[i].get("claude_A")=="faithful"]
print("faithful 子集:", len(sub))
INSTR="""An automatic error generator flagged RAG answer sentences as hallucinations and gave its reasoning. VERIFY each INDEPENDENTLY against the SOURCE — do NOT trust the generator; it may cite evidence not actually in the source or give invalid reasoning. For each ITEM decide the candidate sentence's status vs the source:
- "faithful": actually supported by the source (generator is wrong)
- "conflict": actually contradicts the source
- "baseless": actually asserts specifics not present in the source
Be strict and source-grounded. Return ONLY a JSON array: [{"k":N,"verdict":"faithful|conflict|baseless"}, ...]"""
def block(k,r):
    e=E.get(r["id"],{})
    return (f"[ITEM {k}]\n[SOURCE]\n{r['source']}\n\n[CANDIDATE SENTENCE S{r['idx']}]: {r['err_sent']}\n"
            f"[Generator claim] type={e.get('gen_type')}; evidence=\"{e.get('evidence','')}\"; reason=\"{e.get('reason','')}\"\n")
def run_batch(a):
    bi,batch=a
    prompt=INSTR+"\n\n"+"\n".join(block(i+1,r) for i,r in enumerate(batch))
    try:
        out=subprocess.run(["claude","-p",prompt,"--model","opus"],capture_output=True,text=True,timeout=300).stdout
        arr=json.loads(re.search(r"\[.*\]",out,re.S).group(0)); vd={o["k"]:o for o in arr}
        return [{"id":r["id"],"verify":vd.get(i+1,{}).get("verdict")} for i,r in enumerate(batch)]
    except Exception as ex:
        return [{"id":r["id"],"verify":None,"err":str(ex)[:60]} for r in batch]
B=10; batches=[(i,sub[i:i+B]) for i in range(0,len(sub),B)]
res=[]
with ThreadPoolExecutor(5) as ex:
    for r in ex.map(run_batch,batches): res.extend(r)
json.dump(res,open("/tmp/verify_res.json","w"))
halluc=lambda v: v in ("conflict","baseless")
flips=sum(halluc(r["verify"]) for r in res if r["verify"])
ok=sum(r["verify"] is not None for r in res)
print(f"核验完成 {ok}/{len(sub)}")
print(f"原判faithful中, 核验后翻成'真错'的: {flips}/{ok} = {flips/max(1,ok):.3f}")
print(f"→ 修正后总真错率估计: (48+{flips})/200 = {(48+flips)/200:.3f}")
