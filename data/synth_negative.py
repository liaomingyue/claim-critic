"""负样本合成：对一条正样本按指定错误类型做最小改动，并校验。

每种类型有一种或两种合成方法（recipe），记在 meta.recipe 里，难样本重写时沿用：

| 类型 | recipe | 做法 |
| --- | --- | --- |
| missing_condition / unsupported_inference / wrong_attribution | llm | LLM 最小改写 |
| citation_error | swap（规则） | 挂载错误：claim 不动，引用从真正的出处换成同组另一篇对照文献，证据沿用原出处的原文 |
| citation_error | unsourced（LLM） | 无中生有：LLM 加一个所有文献都没有的具体事实，证据为 none |
| term_error | term（规则） | 按术语词典把 claim 里的标准译名换成误译，证据是英文术语所在的原句 |

LLM 方法的校验（任一不满足即丢弃）：证据逐字出现在参考文献中（unsourced 必须是 none）；与原 claim 的 token 级
编辑距离在 [min_edit, max_edit] 之间；输出的 type 与请求一致。规则方法不调模型，标签没有噪声。

类型分配：先看每条正样本能用哪些方法（单引用且有对照文献才能 swap，被引用文献和 claim 里有词典术语才能 term），
类型挑全局用得最少的，citation_error 内部再在 swap / unsourced 之间挑用得少的；示例池每类随机抽 3 条；
每条正样本最多派生 2 条。

用法：python -m data.synth_negative --config configs/data.yaml --batch 1
"""
from __future__ import annotations

import argparse
import random
import re
from collections import Counter

from common.config import load_config
from common.glossary import Glossary, default_glossary
from common.llm import LLM, LLMConfig, map_resumable
from common.output_format import evidence_locatable, normalize, parse_evidence
from common.prompts import TYPE_DEFINITIONS, sample_input
from common.schema import ERROR_TYPES, Sample, cited_ids, cited_refs, read_jsonl

RECIPES = {  # 类型 -> 可用的合成方法
    "missing_condition": ("llm",), "unsupported_inference": ("llm",), "wrong_attribution": ("llm",),
    "citation_error": ("swap", "unsourced"), "term_error": ("term",),
}
LLM_RECIPES = ("llm", "unsourced")
LLM_TYPES = tuple(t for t in ERROR_TYPES if set(RECIPES[t]) & set(LLM_RECIPES))  # 需要示例池的类型

SYNTH_PROMPT = """你要为忠实性检测器构造训练用的错误样本。下面是一条忠实于参考文献的 claim。
请在其中引入一处「{type_name}」类型的错误。

错误类型定义：{type_def}

示例：
{examples}

要求：改动尽量小，错误要隐蔽，读起来仍然通顺；保持 claim 的语言（中文 claim 仍写中文）；{type_rule}
严格按以下格式输出：
<plan>一句话说明在哪里、怎样引入该类型错误</plan>
<claim>改写后的错误 claim</claim>
<evidence>{evidence_rule}</evidence>
<type>{type_name}</type>
<reason>一句话错误原因</reason>

{input}"""

RECIPE_RULES = {
    "unsourced": "这里只做「无中生有」：加入一个具体的事实（数字、结局、亚组发现等），它在所有参考文献里都找不到，"
                 "但读起来像研究报告的内容。",
}
DEFAULT_TYPE_RULE = "不能引入参考文献中完全没提到的新实体。"
EVIDENCE_RULES = {"unsourced": "none"}
DEFAULT_EVIDENCE_RULE = 'ref-编号: "参考文献中能证明它错了的逐字片段"'

REWRITE_SUFFIX = """

你上一次改写的 claim 被检测器识破了。检测器的分析如下：
<detector>{feedback}</detector>
上次的错误 claim：{prev_claim}
请在同一条原始 claim 上重写，换一种更难被识破的改法，仍然是「{type_name}」类型。"""

_TAG = lambda tag: re.compile(rf"<{tag}>(.*?)</{tag}>", re.S)  # noqa: E731
TOKEN_RE = re.compile(r"[一-鿿]|\w+|[^\w\s]")


def tokens(s: str) -> list[str]:
    return TOKEN_RE.findall(s.lower())


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def parse_synth(text: str) -> dict:
    out = {}
    for tag in ("plan", "claim", "evidence", "type", "reason"):
        m = _TAG(tag).search(text)
        out[tag] = m.group(1).strip() if m else None
    out["think"] = out.pop("plan")  # 改写思路：不用 <think> 标签，Claude 的安全分类器会拦截要求输出推理过程的请求
    return out


def validate(parsed: dict, original_claim: str, refs: dict[str, str], want_type: str,
             min_edit: int = 1, max_edit: int = 40, recipe: str = "llm") -> str:
    """返回空串表示通过，否则返回丢弃原因。"""
    if not parsed.get("claim") or not parsed.get("evidence") or not parsed.get("reason"):
        return "missing field"
    if parsed.get("type") != want_type:
        return "type mismatch"
    if recipe == "unsourced":
        if parsed["evidence"].strip().lower() != "none":
            return "unsourced evidence must be none"
    else:
        ref_id, snippet = parse_evidence(parsed["evidence"])
        if not evidence_locatable(ref_id, snippet, refs):
            return "evidence not verbatim"
    d = edit_distance(tokens(original_claim), tokens(parsed["claim"]))
    if not (min_edit <= d <= max_edit):
        return f"edit distance {d}"
    return ""


def build_prompt(pos: dict, type_name: str, examples: list[dict], feedback: dict | None = None,
                 recipe: str = "llm") -> str:
    ex = "\n\n".join(f"原 claim：{e['original']}\n错误 claim：{e['corrupted']}\n原因：{e['reason']}"
                     for e in examples)
    prompt = SYNTH_PROMPT.format(type_name=type_name, type_def=TYPE_DEFINITIONS[type_name], examples=ex,
                                 type_rule=RECIPE_RULES.get(recipe, DEFAULT_TYPE_RULE),
                                 evidence_rule=EVIDENCE_RULES.get(recipe, DEFAULT_EVIDENCE_RULE),
                                 input=sample_input(pos))
    if feedback:
        prompt += REWRITE_SUFFIX.format(feedback=feedback["critic_output"],
                                        prev_claim=feedback["prev_claim"], type_name=type_name)
    return prompt


# ---------- 规则合成 ----------

def sentence_around(text: str, start: int, end: int, max_chars: int = 300) -> str:
    """text[start:end] 所在的整句（按 ". " 切），过长时截成以它为中心的窗口。text 需已 normalize。"""
    a = text.rfind(". ", 0, start)
    a = 0 if a < 0 else a + 2
    b = text.find(". ", end)
    b = len(text) if b < 0 else b + 1
    if b - a > max_chars:
        mid = (start + end) // 2
        a, b = max(a, mid - max_chars // 2), min(b, mid + max_chars // 2)
    return text[a:b].strip()


def applicable_recipes(pos: dict, glossary: Glossary | None = None) -> dict[str, list[str]]:
    """这条正样本能合成哪些类型、每个类型能用哪些方法。"""
    cited = cited_ids(pos)
    g = glossary or default_glossary()
    ok = {"llm", "unsourced"}
    if len(cited) == 1 and len(pos["refs"]) > 1:
        ok.add("swap")
    if cited and g.substitutable(cited_refs(pos), pos["claim"]):
        ok.add("term")
    out = {t: [r for r in RECIPES[t] if r in ok] for t in ERROR_TYPES}
    return {t: rs for t, rs in out.items() if rs}


def applicable_types(pos: dict, glossary: Glossary | None = None) -> list[str]:
    return list(applicable_recipes(pos, glossary))


def synth_citation_error(pos: dict, rng: random.Random) -> dict | None:
    """claim 不动，引用从真正的出处换成另一篇对照文献。证据沿用正样本的证据（出自原出处）。"""
    cited = cited_ids(pos)
    others = [r for r in pos["refs"] if r not in cited]
    if len(cited) != 1 or not others:
        return None
    wrong = rng.choice(others)
    # 被引用的文献排在最前，和 select_context 的顺序一致；否则「被引用的不在第一位」就成了泄露标签的捷径
    refs = {wrong: pos["refs"][wrong], **{r: t for r, t in pos["refs"].items() if r != wrong}}
    return {"claim": pos["claim"], "cited": [wrong], "refs": refs, "evidence": pos["evidence"],
            "reason": f"claim 的内容出自 {cited[0]}，却标注引用了 {wrong}。",
            "meta": {"recipe": "swap", "rule": f"cite {cited[0]} -> {wrong}"}}


def synth_term_error(pos: dict, rng: random.Random, glossary: Glossary | None = None) -> dict | None:
    """按词典把 claim 里一个标准译名换成误译；证据是被引用文献中英文术语所在的原句。"""
    g = glossary or default_glossary()
    refs = cited_refs(pos)
    terms = g.substitutable(refs, pos["claim"])
    if not terms:
        return None
    t = rng.choice(terms)
    wrong = rng.choice(t.wrong)
    a, b = rng.choice(g.zh_spans(t.zh, pos["claim"]))
    claim = pos["claim"][:a] + wrong + pos["claim"][b:]
    for rid, text in refs.items():
        norm = normalize(text)
        m = g.en_match(t, norm)
        if m:
            evidence = sentence_around(norm, m.start(), m.end())
            return {"claim": claim, "cited": cited_ids(pos), "evidence": f'{rid}: "{evidence}"',
                    "reason": f"原文 {m.group(0)} 应译为「{t.zh}」，claim 写成了「{wrong}」。",
                    "meta": {"recipe": "term", "rule": f"{t.zh} -> {wrong}", "term_en": t.en}}
    return None


# ---------- 入口 ----------

def synthesize_one(llm: LLM | None, pos: dict, type_name: str, pool: dict[str, list[dict]],
                   rng: random.Random, cfg: dict, feedback: dict | None = None,
                   recipe: str | None = None) -> dict | None:
    """返回一条负样本 dict，失败返回 None。recipe 缺省：citation_error 取 unsourced（任何正样本都能做），其余取唯一的方法。
    规则方法忽略 feedback：重写时靠 rng 换一个术语 / 换一篇文献。"""
    recipe = recipe or ("unsourced" if type_name == "citation_error" else RECIPES[type_name][0])
    if recipe == "swap":
        out = synth_citation_error(pos, rng)
    elif recipe == "term":
        out = synth_term_error(pos, rng)
    else:
        examples = rng.sample(pool[type_name], min(cfg["n_examples"], len(pool[type_name])))
        prompt = build_prompt(pos, type_name, examples, feedback, recipe)
        parsed = parse_synth(llm.chat_one([{"role": "user", "content": prompt}]))
        if validate(parsed, pos["claim"], pos["refs"], type_name, cfg["min_edit"], cfg["max_edit"], recipe):
            return None
        out = {"claim": parsed["claim"], "cited": cited_ids(pos), "evidence": parsed["evidence"],
               "reason": parsed["reason"], "meta": {"recipe": recipe, "synth_think": parsed["think"]}}
    if out is None:
        return None
    return Sample(id="", split="train", source_group=pos["source_group"], refs=out.get("refs", pos["refs"]),
                  cited=out["cited"],
                  claim=out["claim"], label="hallucination", type=type_name,
                  evidence=out["evidence"], reason=out["reason"], origin="synthetic",
                  parent_id=pos["id"], meta=out["meta"]).to_dict()


def assign_types(positives: list[dict], per_positive: int) -> list[tuple[dict, str, str]]:
    """每条正样本挑 per_positive 个类型（全局用得最少的优先），每个类型再挑用得最少的方法。返回 (pos, type, recipe)。"""
    used: Counter = Counter()
    used_recipe: Counter = Counter()
    out = []
    for pos in positives:
        options = applicable_recipes(pos)
        types = list(options)
        for _ in range(min(per_positive, len(types))):
            t = min(types, key=lambda x: (used[x], ERROR_TYPES.index(x)))
            types.remove(t)
            r = min(options[t], key=lambda x: (used_recipe[x], RECIPES[t].index(x)))
            used[t] += 1
            used_recipe[r] += 1
            out.append((pos, t, r))
    return out


def load_pool(path: str) -> dict[str, list[dict]]:
    pool: dict[str, list[dict]] = {t: [] for t in LLM_TYPES}
    for e in read_jsonl(path):
        pool[e["type"]].append(e)
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--batch", type=int, required=True, help="第几批负样本（1/2/3），决定输出文件与随机种子")
    args = ap.parse_args()
    cfg = load_config(args.config)
    scfg = cfg["synth_negative"]
    llm = LLM(LLMConfig.from_dict(cfg["generator"]))
    pool = load_pool(scfg["example_pool"])
    rng = random.Random(1000 + args.batch)

    positives = [p for p in read_jsonl(cfg["label_positive"]["out"]) if not p.get("_dropped")]
    rng.shuffle(positives)
    positives = positives[: scfg["positives_per_batch"]]

    items = [{"id": f"{pos['id']}-b{args.batch}-{t}", "pos": pos, "type": t, "recipe": r}
             for pos, t, r in assign_types(positives, scfg["per_positive"])]
    print(f"requested: {dict(Counter((i['type'], i['recipe']) for i in items))}")

    def run(item: dict) -> dict | None:
        s = synthesize_one(llm, item["pos"], item["type"], pool, random.Random(item["id"]), scfg,
                           recipe=item["recipe"])
        if s:
            s["id"] = item["id"]
            s["meta"]["batch"] = args.batch
        return s

    map_resumable(items, run, scfg["out"].format(batch=args.batch), concurrency=llm.cfg.concurrency)


if __name__ == "__main__":
    main()
