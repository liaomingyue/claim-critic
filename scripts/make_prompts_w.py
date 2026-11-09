"""按类型加权的生成器 GRPO 提示（2026-11-09 用户同意：critic 弱且能造难的类型多放，造难就变类型的少放）。
不含末尾 64 个试采提示；矛盾 / 添加事实只留约 1/3；范围偏移 / 近义 / 拼接 / 张冠李戴各放两份。确定性（固定种子）。
  python scripts/make_prompts_w.py   # outputs/rt/gen_v2_grpo/prompts.jsonl → prompts_w.jsonl
"""
import collections
import json
import random

O = "outputs/rt/gen_v2_grpo/"
rows = [json.loads(l) for l in open(O + "prompts.jsonl", encoding="utf-8")]
keep_all = {"certainty_or_scope_shift", "near_synonym_shift", "merged_across_passages", "wrong_attribution",
            "vague_evaluation_or_addon", "unsupported_inference", "number_or_date"}
boost = {"certainty_or_scope_shift", "near_synonym_shift", "merged_across_passages", "wrong_attribution"}
rng = random.Random("gen_v2_grpo|weighted")
out = [r for r in rows[:-64] if r["mechanism"] in keep_all or rng.random() < 0.33]
out = out + [r for r in out if r["mechanism"] in boost]
rng.shuffle(out)
with open(O + "prompts_w.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
c = collections.Counter(r["mechanism"] for r in out)
print("PROMPTS_W", len(out), {k: round(v / len(out), 3) for k, v in c.most_common()})
