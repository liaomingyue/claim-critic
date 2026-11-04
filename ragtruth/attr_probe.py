"""告诉 2B「张冠李戴」是什么，它能不能判对（2026-11-04，用户问）。
验收回答（grpo300all，Claude 标）里 wrong_attribution / merged_across_passages 的错句 42 条（明显 27）+ 全部非错句（去掉拒答，603）。
四种条件：
  base_nodef：2B 底座开思考、不给定义（分开「定义」和「思考」各自的作用）
  ft      ：基线 critic（rm1__mcgold），原 prompt，单字母（对照，同 critic2b_scores）
  ft_def  ：基线 critic，system prompt 里加张冠李戴的定义和核对方法，单字母——微调过的模型听不听新说明
  base_def：2B 底座（不加 LoRA），同样的定义，开思考，先逐个核对人物 / 主体做了什么再给字母——不靠微调、靠推理能不能判出
  9b_def  ：9B 底座，同 base_def（对照：同样说明下大模型能做到多少）
非错句上的误报率一起看：说明可能只是让模型更爱判错。

  python -m ragtruth.attr_probe build
  python -m ragtruth.attr_probe think --inp ... --url http://127.0.0.1:8001/v1 --model base --out x.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

from ragtruth.critic_refuse import REF_RE
from ragtruth.rm_data import RM1_SYSTEM
from ragtruth.task import render_input

D = "outputs/rt/minicpm/grpo300all"
MECH = ("wrong_attribution", "merged_across_passages")
DEF = """Pay special attention to wrong attribution, a common kind of conflict: the sentence assigns an action, statement, opinion, attribute or role to the wrong person, group, organization or thing. Examples: the sentence says A did or said something that the source says B did or said; "his lawyer" when the lawyer belongs to someone else; facts about two different people, events or passages merged as if they were about one. To check, take each person or entity in the sentence, find what the source says that entity did, said or has, and make sure the sentence matches. A wrong attribution is option B (conflict)."""
FT_DEF = RM1_SYSTEM.replace("\n\nAnswer with a single letter", "\n\n" + DEF + "\n\nAnswer with a single letter")
THINK0 = RM1_SYSTEM.replace("Answer with a single letter: A, B or C.", "Think step by step, then end your reply with a final line of the form \"Answer: X\" where X is A, B or C.")
THINK = RM1_SYSTEM.replace("Answer with a single letter: A, B or C.", DEF + "\n\nThink step by step: list each person or entity in the sentence and what the sentence says about them, and check each against the source. Then end your reply with a final line of the form \"Answer: X\" where X is A, B or C.")


def cmd_build(a) -> None:
    ans = {x["id"]: x for f in ("grpo300", "grpo300b") for x in map(json.loads, open(f"outputs/rt/minicpm/{f}/answers.jsonl", encoding="utf-8"))}
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for r in map(json.loads, open(f"{D}/final.jsonl", encoding="utf-8")):
            if REF_RE.search(r["sentence"]) or (r["is_error"] and r["mechanism"] not in MECH):
                continue
            x = ans[r["resp_id"]]
            u = render_input({"source": x["source"], "sentences": x["sentences"], "idx": r["idx"]})
            for tag, sysp in (("ft", RM1_SYSTEM), ("ftdef", FT_DEF), ("think", THINK), ("think0", THINK0)):
                f.write(json.dumps({"id": f"{r['id']}|{tag}", "prompt": [{"role": "system", "content": sysp}, {"role": "user", "content": u}],
                                    "label": tag, "is_error": r["is_error"], "severity": r["severity"], "mechanism": r["mechanism"]}, ensure_ascii=False) + "\n")
                n += 1
    print("ATTR_BUILD", a.out, n, flush=True)


def cmd_think(a) -> None:
    from openai import OpenAI
    cl = OpenAI(base_url=a.url, api_key="x", timeout=1800)
    rows = [r for r in map(json.loads, open(a.inp, encoding="utf-8")) if r["label"] == a.label]

    def run(r: dict) -> dict:
        try:
            resp = cl.chat.completions.create(model=a.model, messages=r["prompt"], max_tokens=a.max_tokens, temperature=0.0,
                                              extra_body={"chat_template_kwargs": {"enable_thinking": True}})
            t = resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            return {"id": r["id"], "letter": None, "err": str(e)[:200]}
        m = re.findall(r"Answer:\s*\**\s*([ABC])", t)
        return {"id": r["id"], "letter": m[-1] if m else None, "tail": t[-300:]}

    with ThreadPoolExecutor(a.concurrency) as ex:
        out = list(ex.map(run, rows))
    with open(a.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("THINK", a.out, len(out), "unparsed", sum(o["letter"] is None for o in out), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "think"])
    ap.add_argument("--out", default=f"{D}/probe_attr.jsonl")
    ap.add_argument("--inp", default=f"{D}/probe_attr.jsonl")
    ap.add_argument("--url")
    ap.add_argument("--model", default="base")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--label", default="think", help="think（带张冠李戴定义）或 think0（只让它思考、不给定义）")
    a = ap.parse_args()
    {"build": cmd_build, "think": cmd_think}[a.cmd](a)


if __name__ == "__main__":
    main()
