import json, collections, random
H="/root/autodl-fs/code/critic_model/outputs/rt/gentarget/harvest_gated.jsonl"
xs=[x for x in (json.loads(l) for l in open(H)) if x.get("p_halluc") is not None and x["p_correct"]>=0.5 and x["orig_p"]<=0.5 and 0.0<=x["p_halluc"]<0.5]
xs.sort(key=lambda x:x["id"]); random.Random(0).shuffle(xs)
seen=set(); per=collections.Counter(); keep=[]
for x in xs:
    if x["sentence"] in seen or per[x["prompt_id"]]>=2: continue
    seen.add(x["sentence"]); per[x["prompt_id"]]+=1; keep.append(x)
    if len(keep)>=2000: break
samp=random.Random(42).sample(keep,200)
for x in samp:
    print(json.dumps({"id":x["id"],"evidence":x.get("evidence",""),"reason":x.get("reason",""),"gen_type":x["type"]},ensure_ascii=False))
