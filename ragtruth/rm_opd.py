"""多维度的事后分类：一次前向连着输出四个字母，每个字母是一个维度，各个位置上分别读选项概率。

opl / opc 要读两遍原文，只给出「哪一种操作」；手段靠词表规则（ragtruth.edit_node），换的是实词或语序时认不出，词表外的同义说法也认不出。
这里把几个维度放进一个模型：
  位置 1  层      A 值 / B 挂接 / C 限定 / D 句间 / E 不是错或清单里没有
  位置 2  操作    A–T，清单里的二十种（位置 1 为 E 时写 Z）
  位置 3  动作    改动的语义方向：replace / add / remove / move / strengthen / weaken / blur（不是错时写 Z）
  位置 4  手段    用的是哪一类词或结构：modal / quantifier / generic_agent / connective / negation / tense / name / number / report / content / structure
节点 = 动作 / 手段。should → may、must → might、conceivably 都是 weaken/modal；12 → several、300 → hundreds of 都是 blur/quantifier。
标签：层和操作是 Claude 独立判的操作（同 rm_opc）。动作和手段全部由 Claude 标（analysis/opd_label，只看原句和改写句）。
原打算规则认得出手段的样本按规则换算（RULE_MAP），抽 150 条让 Claude 也标了一遍：动作一致 44%、手段一致 41%、两项都一致只有 29%。
改写幅度大的句子里，顺带出现的数字、连接词会触发规则，而真正造成错误的改动在别处。所以规则换算的标签不用，那部分也交给 Claude 标了。

  python -m ragtruth.rm_opd      # → outputs/rt/rm/opd__v1/{train,calib,test}.jsonl
"""
import collections
import json
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.core import write_rm
from ragtruth.edit_node import FALLBACK, node
from ragtruth.error_ops import OPS
from ragtruth.rm_opc import LAYER_DESC, LAYER_LETTER, load, user
from ragtruth.rm_ops import row

RT = Path("outputs/rt")
OPLIST = list(OPS)
ACTIONS = ["replace", "add", "remove", "move", "strengthen", "weaken", "blur"]
DEVICES = ["modal", "quantifier", "generic_agent", "connective", "negation", "tense", "name", "number", "report", "content", "structure"]
ACTION_DESC = {"replace": "a specific element is exchanged for a different specific one", "add": "information is added", "remove": "information is removed without changing certainty or scope",
               "move": "existing elements are re-attached, swapped or reordered", "strengthen": "the claim becomes more certain, absolute or general",
               "weaken": "the claim becomes more tentative", "blur": "something specific becomes vague"}
DEVICE_DESC = {"modal": "a modal or hedge word", "quantifier": "an indefinite quantifier or approximator", "generic_agent": "a generic or unnamed agent", "connective": "a connective or linking phrase",
               "negation": "a negation or an antonym", "tense": "tense or aspect", "name": "a proper name", "number": "a number, amount or date", "report": "a report marker",
               "content": "ordinary content words or a longer phrase", "structure": "reordering or re-attaching only"}
OPTS = ["ABCDE", "".join(chr(65 + k) for k in range(len(OPLIST))), "".join(chr(65 + k) for k in range(len(ACTIONS))), "".join(chr(65 + k) for k in range(len(DEVICES)))]
SYSTEM = ("One sentence of an assistant's response was edited. You see the source (the exact input the assistant received), the response with the edited sentence in place, "
          "the ORIGINAL and the EDITED version of that sentence, and the word-level changes. Describe the edit along four dimensions, judged against the source.\n\n"
          "1. Kind of error:\n" + "\n".join(f"{LAYER_LETTER[k]}. {k}: {LAYER_DESC[k]}." for k in LAYER_LETTER)
          + "\nE. none: the edited sentence is still faithful to the source, or the error fits none of the above.\n\n"
          "2. Operation:\n" + "\n".join(f"{chr(65 + k)}. {o}: {OPS[o]['desc']}." for k, o in enumerate(OPLIST))
          + "\n\n3. Action, what the edit does to the claim:\n" + "\n".join(f"{chr(65 + k)}. {a}: {ACTION_DESC[a]}." for k, a in enumerate(ACTIONS))
          + "\n\n4. Device, the kind of words or structure the edit mainly uses:\n" + "\n".join(f"{chr(65 + k)}. {d}: {DEVICE_DESC[d]}." for k, d in enumerate(DEVICES))
          + "\n\nAnswer with four letters separated by spaces, one per dimension in this order. If the first letter is E, answer E Z Z Z.")
# 规则节点的第一部分 → （动作，手段）。带「数字 + 模糊量词」的单独处理（见 from_rule）
RULE_MAP = {"num_swap": ("replace", "number"), "num_drop": ("remove", "number"), "num_add": ("add", "number"),
            "hedge+": ("weaken", "modal"), "hedge-": ("strengthen", "modal"), "vague_qty+": ("blur", "quantifier"), "vague_qty-": ("strengthen", "quantifier"),
            "generic_agent+": ("blur", "generic_agent"), "generic_agent-": ("replace", "generic_agent"), "causal+": ("add", "connective"), "causal-": ("remove", "connective"),
            "contrast_time+": ("add", "connective"), "contrast_time-": ("remove", "connective"), "negation+": ("replace", "negation"), "negation-": ("replace", "negation"),
            "universal+": ("strengthen", "quantifier"), "universal-": ("weaken", "quantifier"), "perfect+": ("strengthen", "tense"), "perfect-": ("weaken", "tense"),
            "report+": ("weaken", "report"), "report-": ("strengthen", "report"), "name_swap": ("replace", "name"), "name_add": ("add", "name")}


def from_rule(n: str) -> tuple[str, str] | None:
    """规则节点换算成（动作，手段）；兜底节点返回 None。"""
    if n in FALLBACK:
        return None
    parts = n.split("|")
    if "vague_qty+" in parts and parts[0].startswith("num_"):
        return "blur", "quantifier"
    return RULE_MAP[parts[0]]


def prompt(x: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user(x)}]


def multi_logprobs(choice, opts: list[str] = OPTS) -> list[dict] | None:
    """按位置读选项的对数概率：第 k 个非空白 token 对应第 k 个维度。位数不够返回 None。"""
    toks = [t for t in ((choice.logprobs.content if choice.logprobs else None) or []) if t.token.strip() not in ("", "<think>", "</think>")]
    if len(toks) < len(opts):
        return None
    out = []
    for t, o in zip(toks, opts):
        lp = {c: -30.0 for c in o + "Z"}
        for c in t.top_logprobs:
            h = c.token.strip().rstrip(".").upper()
            if h in lp:
                lp[h] = max(lp[h], c.logprob)
        out.append(lp)
    return out


def decode(lps: list[dict]) -> dict:
    """四个位置各取概率最高的选项，译回名字。第一位是 E 时其余都是 none。"""
    best = [max(lp, key=lp.get) for lp in lps]
    if best[0] in ("E", "Z"):
        return {"layer": "none", "op": "none", "action": "none", "device": "none"}

    def pick(lp: dict, names: list[str]) -> str:
        b = max((c for c in lp if c != "Z"), key=lp.get)
        return names[ord(b) - 65]

    return {"layer": {v: k for k, v in LAYER_LETTER.items()}[best[0]], "op": pick(lps[1], OPLIST), "action": pick(lps[2], ACTIONS), "device": pick(lps[3], DEVICES)}


def soft_nodes(lps: list[dict]) -> dict:
    """节点（动作 / 手段）上的概率分布，外加 none/none（第一位是 E 的概率）。动作和手段两个位置各自归一后相乘，再乘上「是错」的概率。小于 0.001 的格子不留。"""
    import math

    def norm(lp: dict, keep: str) -> dict:
        e = {c: math.exp(lp[c]) for c in keep}
        s = sum(e.values()) or 1.0
        return {c: v / s for c, v in e.items()}

    p_none = norm(lps[0], OPTS[0])["E"]
    pa, pd = norm(lps[2], OPTS[2]), norm(lps[3], OPTS[3])
    out = {"none/none": p_none}
    for i, a in enumerate(ACTIONS):
        for j, d in enumerate(DEVICES):
            v = (1.0 - p_none) * pa[chr(65 + i)] * pd[chr(65 + j)]
            if v >= 1e-3:
                out[f"{a}/{d}"] = v
    s = sum(out.values())
    return {k: round(v / s, 4) for k, v in out.items()}


def main() -> None:
    rows = load()
    L = {r["uid"]: r for f in ("labels.jsonl", "labels2.jsonl") if (RT / "analysis/opd_label" / f).exists() for r in read_jsonl(RT / "analysis/opd_label" / f)}
    parts, src, agree = collections.defaultdict(list), collections.Counter(), collections.Counter()
    for r in rows:
        if r["op"] not in OPS:
            lab = "E Z Z Z"
        else:
            rule, c = from_rule(node(r["original"], r["sentence"])), L.get(r["uid"])
            if c and rule:   # 两边都有的：量规则换算准不准
                agree["n"] += 1
                agree["action"] += c["action"] == rule[0]
                agree["device"] += c["device"] == rule[1]
                agree["both"] += (c["action"], c["device"]) == rule
            if not c:
                src["dropped: no label"] += 1
                continue
            ad = (c["action"], c["device"])
            src["claude"] += 1
            lab = f"{LAYER_LETTER[OPS[r['op']]['layer']]} {chr(65 + OPLIST.index(r['op']))} {chr(65 + ACTIONS.index(ad[0]))} {chr(65 + DEVICES.index(ad[1]))}"
        parts[r["split"]].append({**row("opd/" + r["uid"], r, prompt(r), lab), "op": r["op"]})
    m = write_rm("opd__v1", parts, 4096)
    print("OPD_V1", json.dumps({k: m[k]["n"] for k in ("train", "calib", "test")}), "action / device labels from", dict(src), "too long", m["too_long_dropped"])
    if agree["n"]:
        print("rule vs Claude on the audit sample: n %d, action %.0f%%, device %.0f%%, both %.0f%%" % (agree["n"], *(100 * agree[k] / agree["n"] for k in ("action", "device", "both"))))
    nodes = collections.Counter(" ".join(r["label"].split()[2:]) for r in parts["train"] if not r["label"].startswith("E"))
    print("nodes in train:", len(nodes), [(f"{ACTIONS[ord(k[0]) - 65]}/{DEVICES[ord(k[2]) - 65]}", v) for k, v in nodes.most_common(12)])


if __name__ == "__main__":
    main()
