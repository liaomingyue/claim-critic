import json, math, sys
def perr(lp,T=1.0):
    xs={k:v/T for k,v in lp.items()}; m=max(xs.values()); e={k:math.exp(v-m) for k,v in xs.items()}; z=sum(e.values())
    return (e.get("B",0)+e.get("C",0))/z
def load(p):
    d={}
    for l in open(p): r=json.loads(l); d[r["id"]]=r
    return d
base=load(sys.argv[1]); aug=load(sys.argv[2]); name=sys.argv[3]
ids=[i for i in base if i in aug]
err=[i for i in ids if base[i]["label"] in ("B","C")]; faith=[i for i in ids if base[i]["label"]=="A"]
bfp_rate=sum(perr(base[i]["lp"])>=0.5 for i in faith)/len(faith)
# 找 aug 阈值 t 使其在忠实上的误报率 == 基线误报率
fp_scores=sorted((perr(aug[i]["lp"]) for i in faith), reverse=True)
k=int(round(bfp_rate*len(faith))); t=fp_scores[k-1] if k>0 else 1.01
bmiss=[i for i in err if perr(base[i]["lp"])<0.5]
recov=sum(perr(aug[i]["lp"])>=t for i in bmiss)
tot_recall_base=sum(perr(base[i]["lp"])>=0.5 for i in err)/len(err)
tot_recall_aug=sum(perr(aug[i]["lp"])>=t for i in err)/len(err)
print(f"[{name}] 匹配基线误报率 {bfp_rate:.3f} → aug阈值 t={t:.3f}")
print(f"  同误报下 整体真错召回: 基线 {tot_recall_base:.3f} → {name} {tot_recall_aug:.3f}")
print(f"  同误报下 基线盲区({len(bmiss)})召回: {recov}/{len(bmiss)} = {recov/max(1,len(bmiss)):.3f}")
