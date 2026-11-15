"""事后分类：给合成样本判实际做成的是哪一种操作（opl 判层，opc 判层内的操作），不看要求的操作。
vLLM 上要有 opl、opc 两个模型。python genops_classify.py '{"opl": T, "opc": T}' ev_st4 ... → outputs/rt/core/samples_ops_<tag>_opc.jsonl
每行：id、layer（value / binding / qualifier / between / none）、p_layer（五个选项的概率）、op（层内概率最高的操作，layer 为 none 时是 none）、p_op。
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from ragtruth.rm_eval import option_logprobs, probs
from ragtruth.rm_opc import LAYER_LETTER, LAYER_OPS, opc_prompt, opl_prompt
from train.gen_reward import NO_THINK

T = json.loads(sys.argv[1])
cl = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="x", timeout=900, max_retries=4)
LETTER_LAYER = {v: k for k, v in LAYER_LETTER.items()}


def P(model, msgs, opts):
    r = cl.chat.completions.create(model=model, messages=msgs, max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=NO_THINK["extra_body"])
    lp = option_logprobs(r.choices[0], opts)
    return probs(lp, T[model]) if lp else None


def one(x):
    x = {**x, "idx": int(x["idx"])}
    out = {"id": x["id"], "layer": None, "op": None}
    p = P("opl", opl_prompt(x), "ABCDE")
    if not p:
        return out
    best = max(p, key=p.get)
    out["p_layer"], out["layer"] = p, LETTER_LAYER.get(best, "none")
    if out["layer"] == "none":
        out["op"] = "none"
        return out
    ops = LAYER_OPS[out["layer"]]
    q = P("opc", opc_prompt(x, out["layer"]), "ABCDEFGH"[:len(ops)])
    if q:
        b = max(q, key=q.get)
        out["op"], out["p_op"] = ops[ord(b) - 65], q[b]
    return out


for tag in sys.argv[2:]:
    xs = [x for x in map(json.loads, open(f"outputs/rt/core/samples_ops_{tag}.jsonl", encoding="utf-8")) if x["why"] == "ok"]
    with ThreadPoolExecutor(96) as ex:
        out = list(ex.map(one, xs))
    open(f"outputs/rt/core/samples_ops_{tag}_opc.jsonl", "w", encoding="utf-8").write("".join(json.dumps(r) + "\n" for r in out))
    print("OPC_DONE", tag, len(out), flush=True)
