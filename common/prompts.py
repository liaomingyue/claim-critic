"""critic 的 prompt。SFT、GRPO、评测、零样本基线、推理全部用这一份，保证训练和评测一致。"""
from __future__ import annotations

from common.glossary import default_glossary
from common.schema import ERROR_TYPES, cited_ids

TYPE_DEFINITIONS = {
    "missing_condition": "前提缺失：丢掉原文的限定条件，把结论推广到原文未覆盖的范围。"
                         "例：原文只针对成年人，claim 写成对所有患者有效。",
    "unsupported_inference": "不合理推理：在原文已有内容上做了原文不支持、医学常识也不许可的推理。"
                             "例：原文说 A 与 B 相关，claim 写成 A 导致 B。",
    "wrong_attribution": "张冠李戴：把另一研究对象、另一研究、另一主体的结论安到当前对象上。"
                         "例：把对照组的结果写成实验组的。",
    "citation_error": "引用错误：claim 的内容不在它标注引用的文献里。两种情况都算这一类："
                      "（1）挂载错误，内容出自另一篇对照文献，例：claim 标注 [ref-2]，内容其实来自 ref-1；"
                      "（2）无中生有，核心事实（数字、结局、发现）在所有参考文献里都找不到，"
                      "例：原文没有报告死亡率，claim 写死亡率下降 30%。",
    "term_error": "术语翻译错误：英文专业术语被译成了另一个意思不同的中文术语。"
                  "例：myocardial infarction（心肌梗死）写成心肌缺血。以输入中的术语表为准。",
}

# 一条 claim 有多处错误时只标一类：先查引用，再查术语，最后查内容
TYPE_PRIORITY = ("citation_error", "term_error", "wrong_attribution", "unsupported_inference", "missing_condition")
assert set(TYPE_PRIORITY) == set(ERROR_TYPES)

CRITIC_SYSTEM = """你是医学综述的忠实性审核员。给定一条 claim、它标注引用的参考文献、同组的对照文献和术语表，
判断 claim 是否忠实于参考文献。

幻觉 = 任何偏离参考文献的内容。只以参考文献为准，不判断世界事实。
标「被引用」的是 claim 标注的出处；标「对照」的是同组其他文献，claim 没有引用它们。
claim 没有标注引用时，只要内容能在任一参考文献中找到支持即为 faithful。不含具体事实的过渡句判 faithful。

错误类型（五选一）：
{defs}

有多处错误时按以下顺序只报第一类：先看内容在不在被引用的文献里（不在 → citation_error），
再看术语译名（term_error），最后看内容（wrong_attribution > unsupported_inference > missing_condition）。

严格按以下格式输出，不要输出其他内容：
<think>你的分析</think>
<verdict>hallucination 或 faithful</verdict>
<evidence>ref-编号: "从该参考文献中逐字复制的片段"</evidence>
<type>{types} 之一；faithful 时写 none</type>
<reason>一句话说明错在哪里；faithful 时可省略本行</reason>

evidence 规则：faithful 时引被引用文献中支持 claim 的原文；citation_error 时引内容真正出处那篇（对照文献）的原文，
所有文献都没有出处时写 <evidence>none</evidence>；其余类型引能说明 claim 错了的原文。""".format(
    defs="\n".join(f"- {t}：{TYPE_DEFINITIONS[t]}" for t in ERROR_TYPES), types="、".join(ERROR_TYPES))


# 用 Claude 当判定模型（label_positive）时的 system prompt：Claude 的安全分类器会拦截「把推理过程写进 <think> 输出」
# 的请求（2026-10-23 实测，Opus 5.5 / claude -p），所以把这一项换成简短的判定依据 <analysis>，
# 拿到输出后再换回 <think>，后面的解析、证据校验与 critic 完全一致。critic 自己的训练格式不变。
JUDGE_SYSTEM = CRITIC_SYSTEM.replace("<think>你的分析</think>", "<analysis>2-4 句判定依据</analysis>")


def judge_messages(s: dict) -> list[dict]:
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": sample_input(s)}]


def judge_to_critic_output(text: str) -> str:
    return text.replace("<analysis>", "<think>").replace("</analysis>", "</think>")

def render_input(refs: dict[str, str], claim: str, cited: list[str] | None = None,
                 terms: list | None = None) -> str:
    """cited 为 None 时视为 refs 全部被引用；terms 为 common.glossary.Term 列表。"""
    cited = list(refs) if cited is None else cited
    lines = ["[references]"]
    for rid, text in refs.items():
        lines.append(f"({rid}) [{'被引用' if rid in cited else '对照'}] {text}")
    if terms:
        lines.append("[glossary]")
        lines += [f"{t.en} = {t.zh}" for t in terms]
    lines += ["[claim]", claim, "[cited]", ", ".join(cited) if cited else "无"]
    return "\n".join(lines)


def sample_input(s: dict) -> str:
    """从样本 dict 渲染 critic 输入，自动附带术语表。所有调用方都走这里。"""
    return render_input(s["refs"], s["claim"], cited_ids(s), default_glossary().hits(s["refs"], s["claim"]))


def critic_messages(s: dict) -> list[dict]:
    """s 至少含 refs、claim，可选 cited。"""
    return [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": sample_input(s)},
    ]
