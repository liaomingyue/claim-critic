import json, collections, random
H="/root/autodl-fs/code/critic_model/outputs/rt/gentarget/harvest_gated.jsonl"
# 复现 critic_harvest gA 的筛选: band[0,0.5) gate0.5 min_nat0, per_prompt<=2, 句子去重, seed0 shuffle, cap2000
xs=[x for x in (json.loads(l) for l in open(H))
    if x.get("p_halluc") is not None and x["p_correct"]>=0.5 and x["orig_p"]<=0.5
    and (x.get("p_natural") or 0)>=0.0 and 0.0<=x["p_halluc"]<0.5]
xs.sort(key=lambda x:x["id"])
rng=random.Random(0); rng.shuffle(xs)
seen=set(); per=collections.Counter(); keep=[]
for x in xs:
    if x["sentence"] in seen or per[x["prompt_id"]]>=2: continue
    seen.add(x["sentence"]); per[x["prompt_id"]]+=1; keep.append(x)
    if len(keep)>=2000: break
print("kept total:", len(keep))
# 抽 200(固定种子)
rng2=random.Random(42); samp=rng2.sample(keep, 200)
with open("/root/eval200.jsonl","w") as f:
    for x in samp:
        resp=[f"S{i}: {t}" for i,t in enumerate(x["context"],1)]
        f.write(json.dumps({"id":x["id"],"source":x["source"],"idx":x["idx"],
            "response_err":resp,
            "err_sent":x["context"][x["idx"]-1],
            "orig_sent":x["sentences_orig"][x["idx"]-1],
            "gen_type":x["type"], "task":x["task_type"],
            "p_halluc":round(x["p_halluc"],3),"p_correct":round(x["p_correct"],3),"p_natural":round(x.get("p_natural") or 0,3)},ensure_ascii=False)+"\n")
print("wrote 200")
