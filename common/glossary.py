"""医学术语词典（configs/glossary.tsv）：英文术语 -> 标准中文译名 + 常见误译。

三处用到，保证训练和推理看到的是同一份词典：
- data/synth_negative.py：term_error 负样本按词典把标准译名换成误译（规则合成，标签无噪声）；
- common/prompts.py：critic 输入里附上参考文献中出现的术语及标准译名，模型对照词典判断，不靠记忆；
- infer/judge.py：规则预检，claim 里出现误译且没有标准译名时直接提示。

中文匹配会屏蔽「更长的词典词」：claim 写「新辅助化疗」时，不算出现了「辅助化疗」。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "configs" / "glossary.tsv"


@dataclass(frozen=True)
class Term:
    en: str
    zh: str
    wrong: tuple[str, ...]


def _zh_re(s: str) -> re.Pattern:
    """中文术语的匹配式，字符间允许空白（「2 型糖尿病」也能匹配「2型糖尿病」）。"""
    return re.compile(r"\s*".join(re.escape(ch) for ch in s if not ch.isspace()), re.I)


class Glossary:
    def __init__(self, terms: list[Term]):
        self.terms = terms
        self._en = {t.en: re.compile(rf"(?<![\w-]){re.escape(t.en)}s?(?![\w-])", re.I) for t in terms}
        self._zh_all = sorted({z for t in terms for z in (t.zh, *t.wrong)}, key=len, reverse=True)

    @classmethod
    def load(cls, path: str | Path) -> "Glossary":
        terms = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            en, zh, wrong = (line.split("\t") + ["", ""])[:3]
            terms.append(Term(en.strip(), zh.strip(), tuple(w.strip() for w in wrong.split("|") if w.strip())))
        return cls(terms)

    # ---------- 匹配 ----------

    def en_in(self, term: Term, text: str) -> bool:
        return bool(self._en[term.en].search(text))

    def en_match(self, term: Term, text: str) -> re.Match | None:
        return self._en[term.en].search(text)

    def zh_spans(self, s: str, text: str) -> list[tuple[int, int]]:
        """s 在 text 中的出现位置，去掉落在更长词典词内部的那些。"""
        longer = [z for z in self._zh_all if len(z) > len(s) and s in z]
        blocked = [m.span() for z in longer for m in _zh_re(z).finditer(text)]
        return [m.span() for m in _zh_re(s).finditer(text)
                if not any(a <= m.start() and m.end() <= b for a, b in blocked)]

    def zh_in(self, s: str, text: str) -> bool:
        return bool(self.zh_spans(s, text))

    # ---------- 对外接口 ----------

    def hits(self, refs: dict[str, str], claim: str, limit: int = 15) -> list[Term]:
        """参考文献中出现的术语。与 claim 有关（标准译名或误译出现在 claim 里）的排前面。

        正样本和 term_error 负样本按同一规则取词条，术语表本身不泄露标签。
        """
        text = "\n".join(refs.values())
        found = [t for t in self.terms if self.en_in(t, text)]
        related = [t for t in found if self.zh_in(t.zh, claim) or any(self.zh_in(w, claim) for w in t.wrong)]
        return (related + [t for t in found if t not in related])[:limit]

    def mistranslations(self, refs: dict[str, str], claim: str,
                        context: dict[str, str] | None = None) -> list[tuple[Term, str]]:
        """规则预检：参考文献里有英文术语，claim 里出现误译、且没有标准译名。

        refs 传 claim 引用的文献（术语从这里取）；context 传全部参考文献（含对照文献），默认同 refs。
        「误译」本身若是 context 里另一个术语的标准译名（文献里同时有 cohort study 和
        randomized controlled trial，claim 写「队列研究」），说明 claim 在说另一件事，不算误译。
        """
        text = "\n".join(refs.values())
        present = [t for t in self.terms if self.en_in(t, text)]
        ctx = "\n".join((context or refs).values())
        legit = {t.zh for t in self.terms if self.en_in(t, ctx)}
        out = []
        for t in present:
            if self.zh_in(t.zh, claim):
                continue
            out += [(t, w) for w in t.wrong if w not in legit and self.zh_in(w, claim)]
        return out

    def substitutable(self, refs: dict[str, str], claim: str) -> list[Term]:
        """可以合成 term_error 的词条：英文出现在参考文献、标准译名出现在 claim、有误译可换。"""
        text = "\n".join(refs.values())
        return [t for t in self.terms if t.wrong and self.en_in(t, text) and self.zh_in(t.zh, claim)]


@lru_cache(maxsize=None)
def _load(path: str) -> Glossary:
    return Glossary.load(path) if Path(path).exists() else Glossary([])


def default_glossary() -> Glossary:
    """环境变量 CRITIC_GLOSSARY 可换词典路径；文件不存在时为空词典。"""
    return _load(os.environ.get("CRITIC_GLOSSARY", str(DEFAULT_PATH)))
