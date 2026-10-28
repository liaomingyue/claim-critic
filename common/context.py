"""给 claim 组装参考文献上下文：被引用的文献 + 同组里最相关的 k 篇对照文献。

只给被引用的文献时，critic 找不到「内容其实在另一篇里」的真正出处，引用错误里的挂载错误和
无中生有就分不开、也给不出证据，所以每条样本都带 k 篇未被引用的对照文献。训练数据（data/split_claims.py）和
推理（infer/judge.py）都用这一个函数，保证 critic 训练时和上线时看到的上下文同分布。

相关度：claim 是中文、文献是英文，直接比词没用。这里只用跨语言也能对上的线索打分：
数字、claim 里保留的英文缩写/药名，以及术语词典把中文术语映射回去的英文词。
需要更准时可以换成多语言向量检索，接口不变。
"""
from __future__ import annotations

import re

from common.glossary import Glossary, default_glossary

_KEY_RE = re.compile(r"\d+(?:\.\d+)?|[a-z][a-z0-9-]{2,}", re.I)
_STOP = {"the", "and", "with", "was", "were", "for", "that", "this", "from", "than", "not", "are", "had",
         "has", "have", "who", "which", "after", "group", "patients", "study", "compared", "versus"}


def _keys(text: str) -> set[str]:
    return {k.lower() for k in _KEY_RE.findall(text) if k.lower() not in _STOP}


def claim_keys(claim: str, glossary: Glossary | None = None) -> set[str]:
    g = glossary or default_glossary()
    keys = _keys(claim)
    for t in g.terms:
        if g.zh_in(t.zh, claim) or any(g.zh_in(w, claim) for w in t.wrong):
            keys |= _keys(t.en)
    return keys


def select_context(claim: str, cited: list[str], group_refs: dict[str, str], k: int = 2,
                   glossary: Glossary | None = None) -> dict[str, str]:
    """返回有序 dict：被引用的文献在前（保持引用顺序），再接相关度最高的 k 篇未引用文献。"""
    ck = claim_keys(claim, glossary)
    others = [rid for rid in group_refs if rid not in cited]
    others.sort(key=lambda rid: -len(ck & _keys(group_refs[rid])))  # 稳定排序，同分保持原顺序
    out = {rid: group_refs[rid] for rid in cited if rid in group_refs}
    for rid in others[:k]:
        out[rid] = group_refs[rid]
    return out
