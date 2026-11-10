"""L1 复现核对：把重算出的 json 和仓库里的 results/<名字>.json 逐字段比较。
  python -m scripts.check_repro <重算的 json> results/<名字>.json
完全一致时打印 REPRO_MATCH；数字（含 bootstrap 区间）都应逐位相同，因为种子是固定的。
"""
import json
import sys


def flat(d, p=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flat(v, f"{p}/{k}")
    else:
        yield p, d


a = dict(flat(json.load(open(sys.argv[1], encoding="utf-8"))))
b = dict(flat(json.load(open(sys.argv[2], encoding="utf-8"))))
bad = [k for k in b if a.get(k) != b[k]]
for k in bad[:10]:
    print("DIFF", k, a.get(k), b[k])
print(f"{len(b)} fields, {len(bad)} mismatches")
print("REPRO_MATCH" if not bad else "REPRO_DIFF")
