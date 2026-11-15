"""把错误操作清单（ragtruth/error_ops.py）和每种操作的真实例句（configs/rt/op_examples.json）导出成一页文档：docs/error_cards.md。

  python -m scripts.dump_cards
"""
import json
from pathlib import Path

from ragtruth.error_ops import CARD, LAYERS, OPS, card

LAYER_ZH = {"value": "值", "binding": "挂接", "qualifier": "限定", "between": "句间"}
PRE_ZH = {"number": "句子里有数字、日期或数量词", "has_number": "句子里有数字", "hedge": "句子里有表示推测或约略的词", "reported": "句子里有转述标记（说、据称、声称）",
          "plan": "句子说的是计划、打算或预期", "owner": "句子里写明了某样东西是谁的"}


def main() -> None:
    ex = json.loads(Path("configs/rt/op_examples.json").read_text(encoding="utf-8"))
    out = ["# 错误操作卡片", "",
           "造错用的错误体系：四层二十种操作，每种操作一张卡片。给错误合成模型出题时，把对应的卡片整张写进提示词。",
           "这一页由 `python -m scripts.dump_cards` 从代码（`ragtruth/error_ops.py`）生成，卡片的原文是英文，因为它直接进提示词。", "",
           "分层的依据是“错的东西从哪来”：", "",
           "| 层 | 含义 | 操作 |", "|---|---|---|"]
    for k in LAYER_ZH:
        ops = [o for o in OPS if OPS[o]["layer"] == k]
        out.append(f"| {LAYER_ZH[k]} | {LAYERS[k]} | " + "、".join(f"[{OPS[o]['zh']}](#{o})" for o in ops) + " |")
    out += ["", "每张卡片有五部分：", "",
            "- **做法**：这种操作对句子做什么。",
            "- **适用条件**：句子要有什么才做得成。没有数字的句子造不出数字错，这类任务出题前就筛掉。",
            "- **为什么算错**：改完之后，句子和原文哪里对不上。",
            "- **怎么检查**：核实时看什么。",
            "- **例子**：两对真实的句子，一句忠于原文，一句是模型自己写错的，两句只差这一处。", ""]
    for k in LAYER_ZH:
        out += [f"## {LAYER_ZH[k]}层", "", f"{LAYERS[k]}。", ""]
        for o in (o for o in OPS if OPS[o]["layer"] == k):
            v = OPS[o]
            need, wrong = CARD[o]
            out += [f'<a id="{o}"></a>', "", f"### {v['zh']}（`{o}`）", "",
                    "| | |", "|---|---|",
                    f"| 做法 | {v['desc']} |",
                    f"| 适用条件 | {need} |",
                    f"| 为什么算错 | {wrong} |",
                    f"| 怎么检查 | {v['check']} |",
                    f"| 出题前的规则 | {PRE_ZH.get(v.get('pre') or '', '无，任何句子都可以出这道题')} |", ""]
            if v.get("examples"):
                out += ["改法举例：", ""] + [f"- {e}" for e in v["examples"]] + [""]
            if ex.get(o):
                out += ["真实例句：", "", "| 忠于原文 | 写错的 |", "|---|---|"]
                out += [f"| {e['faithful'].strip().replace('|', '/')} | {e['error'].strip().replace('|', '/')} |" for e in ex[o]]
                out.append("")
            out += ["<details><summary>写进提示词的卡片原文</summary>", "", "```", card(o, ex.get(o, [])), "```", "", "</details>", ""]
    Path("docs/error_cards.md").write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("wrote docs/error_cards.md", len(out), "lines,", len(OPS), "cards")


if __name__ == "__main__":
    main()
