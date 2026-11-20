"""用重训的把关模型给合成样本打分：sev4（真错 / 明显错）、nat4（自然）、opm（做的是不是要求的操作）、critic。"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from ragtruth.rm_data import RM1_SYSTEM
from ragtruth.rm2_v3 import prompt as nat_prompt
from ragtruth.rm_eval import option_logprobs, probs
from ragtruth.rm_ops import opm_prompt
from ragtruth.rm_disc import prompt as disc_prompt
from ragtruth.sev_gen import prompt_dne
from ragtruth.task import render_input
from train.gen_reward import NO_THINK
T = json.loads(sys.argv[1]); tags = sys.argv[2:]
cl = OpenAI(base_url=os.environ.get("GENOPS_URL", "http://127.0.0.1:8000/v1"), api_key=os.environ.get("VLLM_API_KEY") or "x", timeout=900, max_retries=4)   # GENOPS_URL：打到推理机的 vLLM
def P(model, msgs, opts):
    r = cl.chat.completions.create(model=model, messages=msgs, max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=NO_THINK["extra_body"])
    lp = option_logprobs(r.choices[0], opts)
    return probs(lp, T[model]) if lp else None
def one(x):
    x = {**x, "idx": int(x["idx"])}; out = {"id": x["id"]}
    p = P("sev4", prompt_dne(x), "ABC"); out["p_err"] = (p["A"] + p["B"]) if p else None; out["p_clear"] = p["A"] if p else None
    p = P("nat4", nat_prompt(x), "AB"); out["p_nat"] = p["A"] if p else None
    p = P("opm", opm_prompt(x, x["op"]), "AB"); out["p_op"] = p["A"] if p else None
    p = P("disc", disc_prompt(x["source"], x["context"], x["idx"]), "AB"); out["p_disc"] = p["A"] if p else None
    p = P("rm1", [{"role": "system", "content": RM1_SYSTEM}, {"role": "user", "content": render_input({"source": x["source"], "sentences": x["context"], "idx": x["idx"]})}], "ABC"); out["p_edit"] = (p["B"] + p["C"]) if p else None
    return out
for tag in tags:
    xs = [x for x in map(json.loads, open(f"outputs/rt/core/samples_ops_{tag}.jsonl", encoding="utf-8")) if x["why"] == "ok"]
    with ThreadPoolExecutor(96) as ex: out = list(ex.map(one, xs))
    open(f"outputs/rt/core/samples_ops_{tag}_{os.environ.get('GENOPS_SUF', 'rm4')}.jsonl", "w", encoding="utf-8").write("".join(json.dumps(r) + "\n" for r in out))
    print("SCORE4_DONE", tag, len(out), flush=True)
