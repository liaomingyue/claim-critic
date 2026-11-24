"""rollout 的本地自检（不用 GPU、不调 LLM）：假策略照标准动作逐个调工具，假用户照脚本说话。
检查：① 工具调用写成 Qwen3.5 的格式再解析回来，参数和标准动作一致；② 数据库比对给满分；
③ 算 loss 的 token 解码出来正好是策略写的文字；④ 改错一个写操作参数时数据库比对给 0 分。
  python -m scripts.analysis.cs_rollout_selftest --tok outputs/models/Qwen3.5-2B-tok [--split train] [--limit 0]
"""
from __future__ import annotations

import argparse
import copy
import json

import cs_agent  # noqa: F401
from cs_agent import rollout as R


def call_text(name: str, args: dict) -> str:
    body = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n" for k, v in args.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def corrupt(actions: list) -> list | None:
    """把最后一个写操作的某个 id 参数换成另一个合法格式的 id；没有写操作就返回 None。"""
    acts = copy.deepcopy(actions)
    for a in reversed(acts):
        if a.name.split("_")[0] in ("cancel", "modify", "exchange", "return"):
            for k, v in a.arguments.items():
                if k == "payment_method_id" and isinstance(v, str):
                    a.arguments[k] = "gift_card_0000000"
                    return acts
                if isinstance(v, list) and v and isinstance(v[0], str):
                    a.arguments[k] = v[:-1] if len(v) > 1 else ["0000000000"]
                    return acts
    return None


def main() -> None:
    from transformers import AutoTokenizer
    from tau2.user.user_simulator import UserSimulator
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tok)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    tasks = R.load_tasks(a.split)
    tasks = tasks[: a.limit] if a.limit else tasks

    def fake_user(script):
        it = iter(script)

        def gen(self, message, state):
            from tau2.data_model.message import UserMessage
            m = UserMessage(role="user", content=next(it))
            state.messages.append(m)
            return m, state
        return gen

    stats = {"tasks": 0, "golden_ok": 0, "parse_ok": 0, "mask_ok": 0, "corrupt_tested": 0, "corrupt_zero": 0}
    for t in tasks:
        acts = [x for x in (t.evaluation_criteria.actions or []) if (x.requestor or "assistant") == "assistant"]
        for mode in ("golden", "corrupt"):
            seq = acts if mode == "golden" else corrupt(acts)
            if seq is None:
                continue
            texts = [call_text(x.name, x.arguments) for x in seq] + ["All set. Is there anything else?"]
            it = iter(texts)

            def sample(prompt_ids, max_tokens):
                s = next(it)
                ids = tok(s, add_special_tokens=False)["input_ids"] + [im_end]
                return {"ids": ids, "old_lp": [0.0] * len(ids), "text": s, "finish": "stop"}
            UserSimulator.generate_next_message = fake_user(["Hi, I need help with my order.", "###STOP###"])
            ep = R.run_episode(tok, t, sample, user_llm="none", user_args={})
            if mode == "golden":
                stats["tasks"] += 1
                stats["golden_ok"] += ep["reward"] == 1.0
                got = [c for tr in ep["turns"] for c in tr["calls"]]
                stats["parse_ok"] += [(c["name"], c["arguments"]) for c in got] == [(x.name, x.arguments) for x in seq]
                pol = tok.decode([i for i, m in zip(ep["ids"], ep["lmask"]) if m])
                stats["mask_ok"] += pol.replace("<|im_end|>", "") == "".join(texts)
                if ep["reward"] != 1.0:
                    print("GOLDEN_FAIL", t.id, ep["term"])
            else:
                stats["corrupt_tested"] += 1
                stats["corrupt_zero"] += ep["reward"] == 0.0
    print(json.dumps(stats))
    print("prefix tokens", len(ep["prompt_ids"]), "episode tokens", ep["n_tokens"])


if __name__ == "__main__":
    main()
