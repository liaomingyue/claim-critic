import json, math, sys
def perr(lp, T=1.0):
    xs={k:v/T for k,v in lp.items()}
    m=max(xs.values()); e={k:math.exp(v-m) for k,v in xs.items()}; z=sum(e.values())
    return (e.get("B",0)+e.get("C",0))/z
def load(p):
    d={}
    for l in open(p):
        r=json.loads(l); d[r["id"]]=r
    return d
base=load(sys.argv[1]); 
for name,path in zip(sys.argv[3::2], sys.argv[4::2]):
    aug=load(path)
    ids=[i for i in base if i in aug]
    # 真错 = label in B/C
    err=[i for i in ids if base[i]["label"] in ("B","C")]
    faith=[i for i in ids if base[i]["label"]=="A"]
    bmiss=[i for i in err if perr(base[i]["lp"])<0.5]      # 基线漏判的真错(盲区)
    bcatch=[i for i in err if perr(base[i]["lp"])>=0.5]
    recov=sum(perr(aug[i]["lp"])>=0.5 for i in bmiss)       # Aug 在盲区召回
    retain=sum(perr(aug[i]["lp"])>=0.5 for i in bcatch)     # Aug 保持原本抓到的
    bfp=sum(perr(base[i]["lp"])>=0.5 for i in faith); afp=sum(perr(aug[i]["lp"])>=0.5 for i in faith)
    print(f"[{name}] 真错{len(err)} 忠实{len(faith)}")
    print(f"  基线盲区(漏判真错)={len(bmiss)}  → Aug召回 {recov}/{len(bmiss)} = {recov/max(1,len(bmiss)):.3f}  ★盲区消除率")
    print(f"  基线已抓真错={len(bcatch)}  → Aug保持 {retain}/{len(bcatch)} = {retain/max(1,len(bcatch)):.3f}")
    print(f"  忠实误报: 基线 {bfp}/{len(faith)}={bfp/len(faith):.3f}  Aug {afp}/{len(faith)}={afp/len(faith):.3f}")
