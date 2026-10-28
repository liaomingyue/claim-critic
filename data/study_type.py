"""按标题和摘要判断文献类型（规则，不调用模型）。

返回值：
  meta         系统综述 / Meta 分析
  review       叙述性综述
  clinical     人群研究：RCT、临床试验、队列、病例对照、横断面、真实世界研究
  case_report  病例报告 / 病例系列
  animal       动物实验
  in_vitro     细胞 / 体外实验
  protocol     研究方案（还没有结果）
  other        判断不了（社论、评论、纯方法学等），一律不用

按「排除类优先」的顺序判断：病例报告、研究方案、体外、动物先排除，再看是不是综述，最后才认临床研究。
标题里的信号比摘要可靠，所以综述和动物实验主要看标题。
"""
from __future__ import annotations

import re

_CASE = re.compile(r"\bcase reports?\b|\bcase series\b|\bwe (?:report|present|describe) (?:a|an|the|two|three) "
                   r"(?:\w+ ){0,3}(?:case|patient)s?\b|\ba case of\b")
_PROTOCOL = re.compile(r"\bstudy protocol\b|\bprotocol for\b|\brationale and design\b|\btrial protocol\b")
_IN_VITRO = re.compile(r"\bin vitro\b|\bcell lines?\b|\bcultured\b|\bhepatocytes\b|\bin silico\b")
_ANIMAL = re.compile(r"\b(?:mice|mouse|murine|rats?|rodents?|zebrafish|rabbits?|piglets?|dogs?|canine|"
                     r"porcine|primates?|drosophila|c\. elegans)\b")
_HUMAN = re.compile(r"\b(?:patients?|participants?|adults?|individuals?|subjects|women|men|children|"
                    r"adolescents?|volunteers?|people|persons?|population|residents?|veterans?)\b")
_META = re.compile(r"\bmeta-?analys[ie]s\b|\bsystematic (?:literature )?review\b|\bumbrella review\b|"
                   r"\bpooled analysis\b")
_REVIEW = re.compile(r"\breview\b|\boverview\b|\bstate of the art\b|\bperspectives?\b|\bupdate on\b|"
                     r"\bcurrent (?:evidence|status|concepts)\b")
_DESIGN = re.compile(r"\brandomi[sz]ed\b|\btrials?\b|\bcohort\b|\bcase-control\b|\bcross-sectional\b|"
                     r"\bretrospective(?:ly)?\b|\bprospective(?:ly)?\b|\bregistry\b|\bobservational\b|"
                     r"\breal-world\b|\bpropensity\b|\bfollow-up\b|\benrolled\b|\brecruited\b|\bpost hoc\b")


def classify(title: str, abstract: str, pub_types: list[str] | None = None) -> str:
    t = (title or "").lower()
    text = f"{t} {(abstract or '').lower()}"
    if _CASE.search(text):
        return "case_report"
    if _PROTOCOL.search(t):
        return "protocol"
    if _IN_VITRO.search(t):
        return "in_vitro"
    # 动物词在标题里，或摘要里有动物词却完全不提人群
    if _ANIMAL.search(t) or (_ANIMAL.search(text) and not _HUMAN.search(text)):
        return "animal"
    if _META.search(t) or _META.search(text[:600]):
        return "meta"
    if _REVIEW.search(t) or "review-article" in (pub_types or []):
        return "review"
    if _DESIGN.search(text) and _HUMAN.search(text):
        return "clinical"
    return "other"
