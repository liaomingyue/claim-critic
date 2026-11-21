"""按 9B 金标评测器的分数分层抽配对（两组回答同一原文同一种子），给 Claude 盲标；各层抽样权重另存，事后加权回总体。
  python -m scripts.analysis.strat_pairs --arm base=<目录>:<前缀> --arm new=<目录>:<前缀> --n 120 --out <目录>
分层：这一对两条回答里任一句的最高 P — 高（≥ 0.5）、中（0.1–0.5）、低（< 0.1）；配额 --quota 高:中:低 的份额，层内不够就全取，余量按顺序补给下一层。
同一层内先每个原文最多一对（原文分散），再随机。"""
import argparse, collections, json, random
from pathlib import Path
from common.schema import read_jsonl


def gold(d: str, prefix: str, s: str) -> dict:
    p = collections.defaultdict(float)
    for x in json.load(open(Path(d) / f"{prefix}_{s}_9b.json", encoding="utf-8")):
        k = x["id"].rsplit("-S", 1)[0].split("-", 1)[1]
        p[f"{s}{k}"] = max(p[f"{s}{k}"], x["p"])
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--sets", nargs="+", default=["a", "b", "c"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--quota", default="0.5:0.3:0.2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ok, mx, src = [], [], {}
    for spec in a.arm:
        d, prefix = spec.split("=", 1)[1].rsplit(":", 1)
        keys, m = set(), collections.defaultdict(float)
        for s in a.sets:
            for r in read_jsonl(Path(d) / f"{prefix}_{s}.jsonl"):
                if r.get("finish") == "stop" and r["sentences"]:
                    k = f"{s}{r['id'].split('-', 1)[1]}"; keys.add(k); src[k] = f"{s}:{r['source_id']}"
            for k, v in gold(d, prefix, s).items():
                m[k] = max(m[k], v)
        ok.append(keys); mx.append(m)
    keys = sorted(set.intersection(*ok))
    top = {k: max(m.get(k, 0.0) for m in mx) for k in keys}
    strata = {"high": [k for k in keys if top[k] >= 0.5], "mid": [k for k in keys if 0.1 <= top[k] < 0.5], "low": [k for k in keys if top[k] < 0.1]}
    rng = random.Random(a.seed)
    q = [float(x) for x in a.quota.split(":")]
    want = [round(a.n * x / sum(q)) for x in q]
    pick, info, carry = [], {}, 0
    for (name, ks), w in zip(strata.items(), want):
        w += carry
        ks = sorted(ks); rng.shuffle(ks)
        seen, first, rest = set(), [], []
        for k in ks:
            (rest if src[k] in seen else first).append(k); seen.add(src[k])
        take = (first + rest)[:w]
        carry = w - len(take)
        pick += take
        info[name] = {"N": len(ks), "n": len(take), "weight": round(len(ks) / max(1, len(take)), 4)}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "strat_pairs.json").write_text(json.dumps(sorted(pick)), encoding="utf-8")
    (out / "strat_info.json").write_text(json.dumps({"strata": info, "stratum": {k: n for n, ks in strata.items() for k in ks if k in set(pick)},
                                                     "arms": a.arm, "n_pairs_total": len(keys)}, indent=1), encoding="utf-8")
    print(json.dumps({"pairs_total": len(keys), "picked": len(pick), "strata": info}, ensure_ascii=False))


if __name__ == "__main__":
    main()
