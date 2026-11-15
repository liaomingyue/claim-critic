"""把一处改动归到「手段」节点：相对原句新加或去掉了哪一类词（只用词表和数字、专名规则，不用模型）。

用来认套路：should → may、must → might、加 possibly 都是「hedge+」；12 → several、300 → hundreds of 都是「vague_qty+」。
按具体的词比较时这些各算一种改法，按节点比较时是同一种。训到后期各种操作都在往几个节点上收：
要求改数字，起点一半是换成另一个数（num_swap），后来三分之二带上了 vague_qty+；要求扩大范围，起点多是 all / every（universal+），后来近半是 vague_qty+。
节点名：<类>+ 表示新加了这类词，<类>- 表示去掉了；数字和专名另有 swap / add / drop。一条样本最多取两个（按下面的顺序）。
"""
import collections
import re


def _w(s: str) -> frozenset:
    return frozenset(s.split())


LEX = [
    ("hedge", _w("may might could possibly potentially perhaps likely probably appears appear seems seem possible suggest suggests suggested reportedly apparently")),
    ("vague_qty", _w("several many some few numerous various multiple most hundreds thousands dozens least about around nearly approximately roughly over more than")),
    ("generic_agent", _w("someone somebody people others critics experts officials observers one anyone everybody everyone")),
    ("causal", _w("because since due therefore thus hence so leading led causing caused resulting result consequently")),
    ("contrast_time", _w("but however although though despite while whereas after before then later subsequently meanwhile")),
    ("negation", _w("not no never none without neither nor failed refused denied cannot")),
    ("universal", _w("all every always entire entirely any completely definitely certainly must will fully")),
    ("perfect", _w("has have had been already completed finished now")),
    ("report", _w("said says say according claims claim claimed alleged allegedly reported reports believe believes suspect suspects told stated")),
]
NUMW = _w("one two three four five six seven eight nine ten eleven twelve twenty thirty forty fifty hundred thousand million billion half")
TOK = re.compile(r"[A-Za-z]+(?:'[a-z]+)?|\d[\d,.]*")


def node(original: str, sentence: str) -> str:
    to, ts = TOK.findall(original), TOK.findall(sentence)
    co, cs = collections.Counter(w.lower() for w in to), collections.Counter(w.lower() for w in ts)
    A, R = set(cs - co), set(co - cs)
    out = []

    def isnum(w: str) -> bool:
        return w[0].isdigit() or w in NUMW

    na, nr = any(map(isnum, A)), any(map(isnum, R))
    if na or nr:
        out.append("num_swap" if na and nr else "num_drop" if nr else "num_add")
    for name, lex in LEX:
        if A & lex:
            out.append(name + "+")
        elif R & lex:
            out.append(name + "-")
    cap_a = {w for i, w in enumerate(ts) if i > 0 and w[0].isupper() and w.lower() in A}
    cap_r = {w for i, w in enumerate(to) if i > 0 and w[0].isupper() and w.lower() in R}
    if cap_a:
        out.append("name_swap" if cap_r else "name_add")
    if not out:
        out.append("word_swap" if A and R else "word_add" if A else "word_drop" if R else "reorder")
    return "|".join(out[:2])
