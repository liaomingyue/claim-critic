import json, subprocess, re, sys
from concurrent.futures import ThreadPoolExecutor

rows=[json.loads(l) for l in open("/tmp/eval200.jsonl")]
BATCH=10
INSTR = """You are an expert RAG hallucination auditor. For each ITEM you are given SOURCE passages and a RESPONSE. One sentence position has TWO candidate versions (A and B). Judge EACH version strictly against the SOURCE:
- "faithful": fully supported by the source
- "conflict": contradicts the source
- "baseless": asserts specifics/claims not present in the source (unsupported)
Judge only the single candidate sentence against the source (use the rest of the response only for context). Be strict but fair: paraphrase that stays within the source is faithful; added specifics not in the source are baseless; wrong facts are conflict.
Return ONLY a JSON array, one object per ITEM, no prose:
[{"k":<item number>,"A":"faithful|conflict|baseless","B":"faithful|conflict|baseless"}, ...]"""

def block(k, r):
    resp="\n".join(r["response_err"][:r["idx"]-1] + [f"S{r['idx']}: <<CANDIDATE>>"] + r["response_err"][r["idx"]:])
    return (f"[ITEM {k}]\n[SOURCE]\n{r['source']}\n\n[RESPONSE] (S{r['idx']} is the candidate position)\n{resp}\n\n"
            f"Candidate A for S{r['idx']}: {r['err_sent']}\n"
            f"Candidate B for S{r['idx']}: {r['orig_sent']}\n")

def run_batch(args):
    bi, batch = args
    prompt = INSTR + "\n\n" + "\n".join(block(i+1, r) for i,r in enumerate(batch))
    try:
        out = subprocess.run(["claude","-p",prompt,"--model","opus"],
                             capture_output=True, text=True, timeout=300).stdout
        m = re.search(r"\[.*\]", out, re.S)
        arr = json.loads(m.group(0))
        vd = {o["k"]: o for o in arr}
        res=[]
        for i,r in enumerate(batch):
            o = vd.get(i+1, {})
            res.append({**r, "claude_A":o.get("A"), "claude_B":o.get("B")})
        return res
    except Exception as e:
        return [{**r, "claude_A":None, "claude_B":None, "err":str(e)[:80]} for r in batch]

batches=[(i, rows[i:i+BATCH]) for i in range(0,len(rows),BATCH)]
allres=[]
with ThreadPoolExecutor(5) as ex:
    for res in ex.map(run_batch, batches):
        allres.extend(res); print(f"done {len(allres)}/{len(rows)}", flush=True)
with open("/tmp/verdicts.jsonl","w") as f:
    for r in allres: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print("WROTE verdicts")
