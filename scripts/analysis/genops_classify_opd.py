"""多维度事后分类模型（vLLM 上名为 opd）的评测和使用。

  python genops_classify_opd.py eval                      # 留出测试集上各维度的准确率 → outputs/rt/rm/opd__v1/eval/report_multi.json
  python genops_classify_opd.py classify ev_st4 ...       # 给合成样本分类 → outputs/rt/core/samples_ops_<tag>_opd.jsonl（id、layer、op、action、device）
  python genops_classify_opd.py base ev_base_free         # 起点模型各操作的节点（动作 / 手段）占比 → configs/rt/op_node_base_opd.json，只算真错的样本
"""
import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from ragtruth.error_ops import OPS
from ragtruth.rm_opd import ACTIONS, DEVICES, OPLIST, decode, multi_logprobs, prompt
from ragtruth.rm_opc import LAYER_LETTER
from train.gen_reward import NO_THINK

cl = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="x", timeout=900, max_retries=4)


def call(msgs: list[dict]) -> dict | None:
    r = cl.chat.completions.create(model="opd", messages=msgs, max_tokens=12, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=NO_THINK["extra_body"])
    lps = multi_logprobs(r.choices[0])
    return decode(lps) if lps else None


def gold(label: str) -> dict:
    a = label.split()
    if a[0] == "E":
        return {"layer": "none", "op": "none", "action": "none", "device": "none"}
    return {"layer": {v: k for k, v in LAYER_LETTER.items()}[a[0]], "op": OPLIST[ord(a[1]) - 65], "action": ACTIONS[ord(a[2]) - 65], "device": DEVICES[ord(a[3]) - 65]}


def cmd_eval() -> None:
    D = Path("outputs/rt/rm/opd__v1")
    rows = [json.loads(l) for l in open(D / "test.jsonl", encoding="utf-8")]
    with ThreadPoolExecutor(64) as ex:
        preds = list(ex.map(lambda r: call(r["prompt"]), rows))
    ok = [(gold(r["label"]), p) for r, p in zip(rows, preds) if p]
    rep = {"n": len(rows), "parsed": len(ok)}
    for k in ("layer", "op", "action", "device"):
        rep[k] = round(sum(g[k] == p[k] for g, p in ok) / len(ok), 4)
    err = [(g, p) for g, p in ok if g["layer"] != "none"]
    rep["errors_n"] = len(err)
    rep["node_on_errors"] = round(sum((g["action"], g["device"]) == (p["action"], p["device"]) for g, p in err) / len(err), 4)
    rep["op_on_errors"] = round(sum(g["op"] == p["op"] for g, p in err) / len(err), 4)
    rep["by_layer"] = {}
    for layer in LAYER_LETTER:
        s = [(g, p) for g, p in err if g["layer"] == layer]
        rep["by_layer"][layer] = {"n": len(s), **{k: round(sum(g[k] == p[k] for g, p in s) / max(1, len(s)), 3) for k in ("layer", "op", "action", "device")}}
    rep["by_op"] = {o: {"n": sum(g["op"] == o for g, _ in err), "acc": round(sum(g["op"] == o == p["op"] for g, p in err) / max(1, sum(g["op"] == o for g, _ in err)), 3)} for o in OPLIST}
    (D / "eval").mkdir(exist_ok=True)
    (D / "eval/report_multi.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("OPD_EVAL", json.dumps({k: v for k, v in rep.items() if k != "by_op"}))


def classify(tag: str) -> list[dict]:
    xs = [x for x in map(json.loads, open(f"outputs/rt/core/samples_ops_{tag}.jsonl", encoding="utf-8")) if x["why"] == "ok"]

    def one(x: dict) -> dict:
        d = call(prompt({**x, "idx": int(x["idx"])}))
        return {"id": x["id"], "want": x["op"], **(d or {"layer": None, "op": None, "action": None, "device": None})}

    with ThreadPoolExecutor(96) as ex:
        out = list(ex.map(one, xs))
    open(f"outputs/rt/core/samples_ops_{tag}_opd.jsonl", "w", encoding="utf-8").write("".join(json.dumps(r) + "\n" for r in out))
    print("OPD_DONE", tag, len(out), flush=True)
    return out


def cmd_base(tag: str) -> None:
    f = Path(f"outputs/rt/core/samples_ops_{tag}_opd.jsonl")
    out = [json.loads(l) for l in open(f, encoding="utf-8")] if f.exists() else classify(tag)
    P = {r["id"]: r for r in map(json.loads, open(f"outputs/rt/core/samples_ops_{tag}_rm4.jsonl", encoding="utf-8"))}
    B = collections.defaultdict(collections.Counter)
    for r in out:
        if r["action"] and r["id"] in P and (P[r["id"]]["p_err"] or 0) >= 0.5:
            B[r["want"]][f"{r['action']}/{r['device']}"] += 1
    base = {op: {"n": sum(c.values()), "share": {k: round(v / sum(c.values()), 4) for k, v in c.items()}} for op, c in B.items()}
    Path("configs/rt/op_node_base_opd.json").write_text(json.dumps(base, indent=0, sort_keys=True), encoding="utf-8")
    print("OPD_BASE", {op: (b["n"], max(b["share"], key=b["share"].get)) for op, b in base.items()})


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        cmd_eval()
    elif sys.argv[1] == "base":
        cmd_base(sys.argv[2])
    else:
        for t in sys.argv[2:]:
            classify(t)
    assert all(o in OPS for o in OPLIST)
