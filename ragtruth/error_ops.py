"""错误操作清单：把「模型会犯、critic 会漏」的忠实性错误整理成一条条能照着做的改写操作。

每条操作有：所属的层、给生成器看的改写说明、核实时要对照原文的什么、能不能做的前提规则、真实错误里的例子。
前提规则只给「限定」层和数字用：这些操作要求句子里先有可改的原料（推测词、限定词、数字），没有就不该派这个任务。
「挂接」层的操作在语法上大多数句子都能做，规则筛不掉什么，不设前提，靠改写说明和后面的核实把关。

用法：
  python -m ragtruth.error_ops                       # 打印清单
  python -m ragtruth.error_ops --sentences F.jsonl   # 每行 {"sentence", "source"}，统计每条操作有多少句子满足前提
"""
import argparse
import collections
import json
import re

LAYERS = {
    "value": "某个成分本身被换掉或添上",
    "binding": "成分都在原文里，但安错了地方",
    "qualifier": "确定程度、范围、条件变了",
    "between": "句子之间的关系，或凭空加上的推论",
}

NUM = re.compile(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|hundred|thousand|million|billion|half|dozen|first|second|third)\b", re.I)
HEDGE = re.compile(r"\b(may|might|could|likely|unlikely|possibly|probably|perhaps|potentially|apparently|appears?|appeared|seems?|seemed|suggests?|suggested|suspected|believed|estimated|thinks?|thought)\b", re.I)
REPORTED = re.compile(r"\b(reportedly|allegedly|alleged|reports? of|according to|said|says|told|stated|claimed|claims|reported|argued|warned|sources?)\b", re.I)
PLAN = re.compile(r"\b(aims? to|plans? to|planned to|set to|expected to|hopes? to|intends? to|seeks? to|seeking to|looking to|trying to|attempt(s|ed|ing)? to|scheduled|proposed|will|would|should)\b", re.I)
QUANT = re.compile(r"\b(some|many|most|several|a few|few|often|sometimes|usually|generally|typically|certain|nearly|almost|about|around|approximately|roughly|at least|up to|on average|partly|mainly|mostly|largely|one of|a number of|in some cases|more than|less than)\b", re.I)


def _nums(t: str) -> set[str]:
    return set(re.findall(r"\d[\d,.:/]*", t))


PRE = {   # 前提规则：句子（和原文）里有没有这条操作要用的原料
    "number": lambda s, src: bool(NUM.search(s)) and len(_nums(src) - _nums(s)) >= 1,
    "has_number": lambda s, src: bool(NUM.search(s)),
    "hedge": lambda s, src: bool(HEDGE.search(s)),
    "reported": lambda s, src: bool(REPORTED.search(s)),
    "plan": lambda s, src: bool(PLAN.search(s)),
    "quantifier": lambda s, src: bool(QUANT.search(s)),
}

# action：add 添上原文没有的内容；delete 去掉原文有的限定；change 换成原文没有的说法；move 只用原文里的材料，但安错了地方
OPS = {
    # ---- 值 ----
    "swap_term": {
        "layer": "value", "zh": "近义替换", "action": "change", "pre": None,
        "desc": "replace one word or short phrase with a similar one that refers to something different (a neighbouring category, a stronger or weaker act), changing nothing else",
        "check": "找到原文里对应的那个词，看换上的词所指是不是同一样东西",
        "examples": ["Replace 'friends' with 'family members'.", "Replace 'nominated for eviction' with 'evict'.", "Replace 'the people of Latin America' with 'Latin American leaders'."],
    },
    "change_number": {
        "layer": "value", "zh": "改数字", "action": "change", "pre": "number",
        "desc": "change a number, date, duration or count to a different value, or turn a count into a proportion or the reverse",
        "check": "在原文里找到这个数，核对数值和单位",
        "examples": ["Change 'around 15' to 'over 15'.", "Change the count '55 Americans' into '55% American'."],
    },
    "add_detail": {
        "layer": "value", "zh": "加事实", "action": "add", "pre": None,
        "desc": "add or substitute one specific detail that the source does not contain (what happened next, where, what kind), stated as plainly as the rest of the sentence",
        "check": "在原文里找这个细节，找不到就是错",
        "examples": ["Replace 'was discharged' with 'resumed work'.", "Add a clause saying some returned to their villages."],
    },
    "flip_outcome": {
        "layer": "value", "zh": "说成相反", "action": "change", "pre": None,
        "desc": "state the opposite of what the source says about an outcome, a stance or whether something happened",
        "check": "在原文里找到这件事的结果或态度，看是不是反了",
        "examples": ["Replace 'stabbed, in stable condition' with 'stabbed to death'.", "Flip '78 Republicans voting for expulsion' to '78 Republicans voting against his removal'."],
    },
    # ---- 挂接 ----
    "change_speaker": {
        "layer": "binding", "zh": "换说话人", "action": "move", "pre": None,
        "desc": "attribute a statement, view or announcement to a different person or body that also appears in the source",
        "check": "在原文里找这句话是谁说的",
        "examples": ["Attribute the attendant's announcement and warning about symptoms to a passenger.", "Turn a USCIS representative's statement into something Jaddou observed."],
    },
    "move_modifier": {
        "layer": "binding", "zh": "修饰语挂错", "action": "move", "pre": None,
        "desc": "move a description, property, role or criticism from the thing it belongs to onto another thing mentioned in the source",
        "check": "在原文里找这个描述说的是哪个对象",
        "examples": ["Move 'in bad shape' from the boat to the man.", "Move the 'amateurish' criticism of Rubio's logo onto Clinton's launch.", "Transfer the author's Kentucky State Senate service to Rand Paul."],
    },
    "reverse_direction": {
        "layer": "binding", "zh": "方向倒置", "action": "move", "pre": None,
        "desc": "reverse the direction of a relation while keeping both sides: swap who did it and who it was done to, swap cause and effect, or swap giver and receiver",
        "check": "在原文里找这个关系的两端，看谁对谁、谁因谁果",
        "examples": ["Swap agent and patient so that Cox knocks over Hawking instead of Hawking knocking over Cox.", "Make Congress the recipient of funding instead of the body that should give funding to USCIS.", "Reverse the causal link so the rule change shortens the game, instead of the shortened game prompting the change."],
    },
    "swap_referent": {
        "layer": "binding", "zh": "所指换对象", "action": "move", "pre": None,
        "desc": "make a pronoun, a short referring phrase or the actor of an action point to a different person or thing from the source than the one the source means",
        "check": "在原文里找这个说法或代词指的是谁、是什么",
        "examples": ["Resolve 'his body' to Hernandez instead of Odin Lloyd.", "Make 'send back to Earth' refer to the delivered cargo instead of the experiments and trash packed for return.", "Replace the actor 'prosecutors' with 'the court'."],
    },
    "change_owner": {
        "layer": "binding", "zh": "所属关系换人", "action": "move", "pre": None,
        "desc": "change whose something is: a relative, a possession, an employee or a work is given to a different person or body from the source",
        "check": "在原文里找这个人或东西属于谁",
        "examples": ["Turn her daughter's two children (her grandchildren) into 'her two children'.", "Change 'Copeland's driver' to 'the club's driver'."],
    },
    "move_number": {
        "layer": "binding", "zh": "数量挂错", "action": "move", "pre": "has_number",
        "desc": "keep a number from the source but make it count or measure a different thing",
        "check": "在原文里找这个数，看它数的是什么",
        "examples": ["Move the age 13 from the time of the incident to the narrator's current age.", "Reattach the 5-mile distance from crash-site-to-campus to university-to-campus."],
    },
    "move_time_place": {
        "layer": "binding", "zh": "时间地点挂错", "action": "move", "pre": None,
        "desc": "keep a time or place from the source but tie it to a different event",
        "check": "在原文里找这个时间或地点，看它属于哪件事",
        "examples": ["Attach the Monday Imax premiere to the film instead of the trailer footage.", "Attach the 12:21 a.m. time of the alarm company's call to police to the alarm's activation instead."],
    },
    "merge_events": {
        "layer": "binding", "zh": "两处并成一句", "action": "move", "pre": None,
        "desc": "take a detail the source gives about one event, rule or object and attach it to a different one described elsewhere in the source",
        "check": "把句子拆成几个说法，分别回原文找，看它们说的是不是同一件事",
        "examples": ["Attach the nominations, which came from Jag's HOH win, to his POV win.", "Attach the old rule's reset-at-Tribal-Council wrinkle to the current rule."],
    },
    # ---- 限定 ----
    "drop_hedge": {
        "layer": "qualifier", "zh": "删改推测词", "action": "delete", "pre": "hedge",
        "desc": "remove or strengthen a hedge so that a possibility, suspicion or estimate reads as settled",
        "check": "在原文里找对应的说法，看有没有 may、likely、疑似这类词",
        "examples": ["Delete the hedge 'may be' so the claim becomes definite.", "Delete 'suspected to be', stating the tsunami origin as fact.", "Replace 'found no reason to doubt' with the stronger 'confirmed'."],
    },
    "drop_reported": {
        "layer": "qualifier", "zh": "删掉「据称」", "action": "delete", "pre": "reported",
        "desc": "remove the marker that something is a report, allegation or one party's claim, so it reads as established fact",
        "check": "在原文里看这件事是事实陈述，还是某一方的说法",
        "examples": ["Delete 'allegedly' so the parking-without-blinker claim is stated as fact.", "Remove the attribution to Defense Minister Kenney, presenting his claim as fact."],
    },
    "plan_as_done": {
        "layer": "qualifier", "zh": "打算说成已发生", "action": "delete", "pre": "plan",
        "desc": "present a plan, aim, proposal, expectation or hypothetical as something that has happened or is certain to happen",
        "check": "在原文里看这件事是已经发生，还是打算、预计、建议",
        "examples": ["Delete 'aims to' so touring in 2018 is stated as certain.", "Drop 'looks set to' so Germany is presented as having already joined.", "Change 'will be looking to win its first title since 1941' into 'secured its first title since 1941'."],
    },
    "generalize": {
        "layer": "qualifier", "zh": "范围放大", "action": "delete", "pre": "quantifier",
        "desc": "widen the scope of a claim: drop a quantifier such as 'many', 'one of' or 'on average', or turn one named case into a group",
        "check": "在原文里看这句话说的是一个、一部分，还是全部",
        "examples": ["Delete 'on average', so the $50+ increase applies to every retiree.", "Drop 'one of' and pluralize 'garrison' to 'garrisons'.", "Generalize one survivor's lament into survivors as a group."],
    },
    "drop_restriction": {
        "layer": "qualifier", "zh": "删条件或限制", "action": "delete", "pre": None,
        "desc": "remove a condition or a restricting phrase so the claim covers more than the source says",
        "check": "在原文里找这句话成立的条件和适用范围",
        "examples": ["Drop the restriction 'from body fat' so that 'no fat loss' becomes 'no weight loss'.", "Delete the restriction 'who deals with infectious disease' so the county seems to have only one doctor overall.", "Delete the condition 'who entered the US by July 31'."],
    },
    # ---- 句子之间 ----
    "add_link": {
        "layer": "between", "zh": "加或换连接词", "action": "move", "pre": None,
        "desc": "connect two statements from the source with a causal, contrastive or temporal connective the source does not use, or replace the source's connective with a different one",
        "check": "在原文里看这两件事之间有没有写明因果、转折或先后",
        "examples": ["Link two separate source sentences with the causal connective 'as'.", "Change 'because he kept winning challenges' to 'despite winning challenges'.", "Prefix the sentence with the contrastive phrase 'Despite the incident', which the source does not contain."],
    },
    "add_conclusion": {
        "layer": "between", "zh": "加结论或评价", "action": "add", "pre": None,
        "desc": "add an interpretation, cause, purpose or evaluation that the source does not state, or turn the source's silence on a point into an explicit statement",
        "check": "在原文里找这个结论或评价，原文没说就是错",
        "examples": ["Turn 'issued a statement that did not reference pressurization issues' into 'stated no such problem existed'.", "Replace the unclear-direction correlation with a claim that marriage improves well-being."],
    },
}


def applicable(sentence: str, source: str) -> list[str]:
    """这句话上能派哪些操作：有前提规则的要满足规则，没有的一律可派。"""
    return [k for k, o in OPS.items() if o["pre"] is None or PRE[o["pre"]](sentence, source)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", default="")
    a = ap.parse_args()
    if not a.sentences:
        for layer, zh in LAYERS.items():
            print(f"\n== {layer}：{zh}")
            for k, o in OPS.items():
                if o["layer"] == layer:
                    print(f"  {k:18s} {o['zh']}  [{o['action']}]  前提：{o['pre'] or '不设'}\n      {o['desc']}")
        return
    rows = [json.loads(l) for l in open(a.sentences, encoding="utf-8")]
    n = collections.Counter(k for r in rows for k, o in OPS.items() if o["pre"] and PRE[o["pre"]](r["sentence"], r["source"]))
    print("sentences", len(rows))
    for k, o in OPS.items():
        if o["pre"]:
            print(f"  {k:18s} {o['zh']}  {100 * n[k] / max(1, len(rows)):.0f}%")


if __name__ == "__main__":
    main()
