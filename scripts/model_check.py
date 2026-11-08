"""模型自检（README「多模型支持」）：一个新模型过了这几项才算支持。

  python -m scripts.model_check --model qwen3.5-9b                       # 只查分词器 / 模板（不需要 GPU）
  python -m scripts.model_check --model qwen3.5-9b --url http://127.0.0.1:8000/v1   # 再查 vLLM：基座生成能停、LoRA 能挂、能返回选项 logprob

检查项：
  template   聊天模板能按配置的参数（如 enable_thinking=False）渲染，含 system / user 内容，以生成提示结尾
  letters    A–F 每个选项字母（含前导空格的变体）是单个 token：打分模型只读答案位置这一个 token 的 logprob
  eos        配置的结束符都在词表里
  share      prompt 共享实现的等价性单测（tests/test_prefix_share*.py，小模型上和整段拼接比）
  generate   （--url）基座回答一个短问题，finish_reason = stop
  lora       （--url）检查用 LoRA（check_adapter，服务名 chk）在 /models 里，RM1 提示下 A / B / C 都拿到 logprob
结果写 outputs/rt/model_check/<名字>.json，任何一项失败退出码 1。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common.models import env, load

SHARE_TEST = {"train.prefix_share_llama": "tests/test_prefix_share_llama.py", "train.prefix_share": "tests/test_prefix_share.py"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="", help="默认本地盘副本 /root/autodl-tmp/models/<目录名>")
    ap.add_argument("--url", default="")
    ap.add_argument("--skip_share", action="store_true")
    a = ap.parse_args()
    m = load(a.model)
    path = a.tokenizer or env(m)["MODEL_LOCAL"]
    res: dict[str, dict] = {}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    try:
        msgs = [{"role": "system", "content": "SYS_MARK"}, {"role": "user", "content": "USER_MARK"}]
        t = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **m["chat_kwargs"])
        no_gen = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False, **m["chat_kwargs"])
        res["template"] = {"ok": "SYS_MARK" in t and t.find("SYS_MARK") < t.find("USER_MARK") and len(t) > len(no_gen), "tail": t[-120:]}
    except Exception as e:   # noqa: BLE001
        res["template"] = {"ok": False, "error": repr(e)}
    lens = {L: [len(tok.encode(v, add_special_tokens=False)) for v in (L, " " + L)] for L in "ABCDEF"}
    res["letters"] = {"ok": all(n[0] == 1 for n in lens.values()), "tokens": lens}
    ids = tok.convert_tokens_to_ids(m.get("eos_tokens", []))
    res["eos"] = {"ok": all(i is not None and i != tok.unk_token_id for i in ids), "ids": ids}
    if not a.skip_share:
        test = SHARE_TEST.get(m["share_impl"])
        if test:
            p = subprocess.run([sys.executable, "-m", "pytest", "-q", test], capture_output=True, text=True)
            res["share"] = {"ok": p.returncode == 0, "test": test, "tail": p.stdout[-300:]}
        else:
            res["share"] = {"ok": True, "note": "share_impl none：不共享 prompt，训练走整段拼接"}

    if a.url:
        from openai import OpenAI
        from ragtruth.rm_data import RM1_SYSTEM
        from ragtruth.rm_eval import option_logprobs
        from ragtruth.task import render_input
        cl = OpenAI(base_url=a.url, api_key="x", timeout=600)
        extra = {"chat_template_kwargs": m["chat_kwargs"]}
        try:
            r = cl.chat.completions.create(model="base", messages=[{"role": "user", "content": "Name the capital of France in one sentence."}],
                                           max_tokens=128, temperature=0.0, extra_body=extra)
            res["generate"] = {"ok": r.choices[0].finish_reason == "stop", "text": r.choices[0].message.content[:200]}
        except Exception as e:   # noqa: BLE001
            res["generate"] = {"ok": False, "error": repr(e)}
        try:
            names = [x.id for x in cl.models.list().data]
            user = render_input({"source": "The museum opens at 9 am.", "sentences": ["The museum opens at 10 am."], "idx": 1})
            r = cl.chat.completions.create(model="chk", messages=[{"role": "system", "content": RM1_SYSTEM}, {"role": "user", "content": user}],
                                           max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=extra)
            lp = option_logprobs(r.choices[0], "ABC")
            res["lora"] = {"ok": "chk" in names and lp is not None and max(lp.values()) > -30, "models": names, "logprobs": lp}
        except Exception as e:   # noqa: BLE001
            res["lora"] = {"ok": False, "error": repr(e)}

    ok = all(v["ok"] for v in res.values())
    out = Path("outputs/rt/model_check"); out.mkdir(parents=True, exist_ok=True)
    (out / f"{m['name']}.json").write_text(json.dumps({"model": m["name"], "ok": ok, **res}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("MODEL_CHECK", m["name"], "OK" if ok else "FAIL", json.dumps({k: v["ok"] for k, v in res.items()}), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
