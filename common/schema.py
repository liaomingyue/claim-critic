"""样本格式与 JSONL 读写。所有阶段的输入输出都是一行一个 JSON 对象。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

LABELS = ("faithful", "hallucination")
# 内容层面 3 类 + 引用错误 + 术语翻译错误。
# citation_error 含两种情况：内容出自另一篇文献（挂载错误，evidence 引真正出处）和
# 哪篇文献都没有（无中生有，evidence 为 none），见 is_unsourced。
ERROR_TYPES = ("missing_condition", "unsupported_inference", "wrong_attribution",
               "citation_error", "term_error")
TYPES = ERROR_TYPES + ("none",)


@dataclass
class Sample:
    id: str
    split: str                      # train | val | test
    source_group: str               # 文献组 id，划分数据集时按组隔离
    refs: dict[str, str]            # ref id -> 参考文献片段：被引用的文献 + 同组检索出的对照文献
    claim: str
    label: str                      # faithful | hallucination
    type: str = "none"              # ERROR_TYPES 之一，faithful 时为 none
    evidence: str = ""              # 'ref-2: "逐字片段"'
    reason: str = ""
    origin: str = "positive"        # positive | synthetic | human
    round: int = 0                  # 难样本挖掘轮次，正样本为 0
    parent_id: str | None = None    # 负样本派生自哪条正样本
    meta: dict = field(default_factory=dict)
    cited: list[str] | None = None  # claim 标注引用的 ref id；None = refs 全部被引用，[] = 句子没有引用

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"bad label {self.label!r} in {self.id}")
        if self.type not in TYPES:
            raise ValueError(f"bad type {self.type!r} in {self.id}")
        if (self.label == "faithful") != (self.type == "none"):
            raise ValueError(f"label/type mismatch in {self.id}: {self.label}/{self.type}")
        if self.cited and not set(self.cited) <= set(self.refs):
            raise ValueError(f"cited ref not in refs in {self.id}: {self.cited}")

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)


def is_unsourced(sample: dict) -> bool:
    """citation_error 中「无中生有」的那种：证据是 none，哪篇参考文献都没有这条内容。"""
    return sample.get("type") == "citation_error" and (sample.get("evidence") or "").strip().lower() == "none"


def cited_ids(sample: dict) -> list[str]:
    """claim 标注引用的 ref id；老数据 / OOD 数据没有 cited 字段时视为 refs 全部被引用。"""
    c = sample.get("cited")
    return list(sample["refs"]) if c is None else list(c)


def cited_refs(sample: dict) -> dict[str, str]:
    """claim 引用的那几篇；句子没有引用时返回全部 refs（内容可以出自任一篇）。"""
    c = [r for r in cited_ids(sample) if r in sample["refs"]]
    return {r: sample["refs"][r] for r in c} if c else dict(sample["refs"])


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_samples(path: str | Path) -> list[Sample]:
    return [Sample.from_dict(d) for d in read_jsonl(path)]
