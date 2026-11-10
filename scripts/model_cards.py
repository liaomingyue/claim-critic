"""每个发布模型的训练数据和训练配置（2026-11-10，用户：我们有很多模型，要给到每个模型具体的训练数据和训练配置）。

读 configs/release_models.yaml（人工整理：用途、数据来源、起点、奖励、重训命令、对应哪张表），
再从数据目录读实际条数 / 标签分布（outputs/rt/rm/<名字>/meta.json）和校准温度（eval/report.json），生成：
  docs/MODELS.md                     总表 + 每个模型一节
  <--cards_dir>/<发布名>/README.md    每个模型的说明（上传 Hugging Face 时替换 PEFT 自带的空模板）
  <--cards_dir>/<发布名>/training.json 同样内容的机器可读版

  python -m scripts.model_cards [--data_root .] [--cards_dir release_cards]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
GROUP = {"critic": "critic（逐句判有没有幻觉）", "rm": "其他打分模型", "rag": "RAG 模型（被训的回答模型）", "gen": "错误生成器"}
LABEL = {"A": "A 有依据", "B": "B 轻微", "C": "C 明显"}


def release_dir(name: str) -> str:
    """登记表里的名字 → 发布包里的 adapters/ 子目录。"""
    first, _, rest = name.partition("/")
    return "adapters/" + first.removesuffix("-mc") + (f"/{rest}" if rest else "")   # 发布包去掉了 -mc 后缀（release_pack.model_name）


def rm_stats(root: Path, name: str) -> dict:
    d = root / "outputs/rt/rm" / name
    out = {}
    if (d / "meta.json").exists():
        m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        out["max_tokens"] = m.get("max_tokens")
        for k in ("train", "calib"):
            if isinstance(m.get(k), dict):
                out[k] = {"n": m[k].get("n"), "label": m[k].get("label")}
        for k in ("base", "added_errors", "added_originals", "harvest", "band", "arm", "labeled", "verify_kept", "merge_added", "threshold"):
            if k in m and m[k] not in (None, 0, ""):
                out.setdefault("build", {})[k] = m[k]
    if (d / "eval/report.json").exists():
        r = json.loads((d / "eval/report.json").read_text(encoding="utf-8"))
        out["temperature"] = r.get("temperature")
        cal = r.get("calib_raw") or {}
        if cal:
            out["calib_auc"] = cal.get("auc")
    return out


def card(m: dict, dfl: dict, root: Path) -> dict:
    meth = m["method"]
    c = {"name": m.get("display", m["name"]), "release_path": release_dir(m["name"]), "group": m["group"],
         "base": m.get("base", dfl["base"]), "method": meth, "purpose": m["purpose"], "used_in": m.get("used_in", "")}
    if meth == "option_rm":
        o = dfl["option_rm"]
        c["config"] = m.get("config", o["config"])
        c["lora"] = o["lora"]
        c["trainer"] = m.get("trainer", o["trainer"])
        c["data"] = [f"outputs/rt/rm/{m['name']}/train.jsonl", f"outputs/rt/rm/{m['name']}/calib.jsonl（只用来拟合温度）"]
        c["data_from"] = m.get("data_from", "")
        c["stats"] = rm_stats(root, m["name"])
        if "train_rows" in m and not c["stats"].get("train"):   # 数据目录没解压时用登记表里的数
            c["stats"]["train"] = {"n": m["train_rows"], "label": m.get("train_label")}
            c["stats"].setdefault("max_tokens", 3072)
        if "temperature" in m and c["stats"].get("temperature") is None:
            c["stats"]["temperature"] = m["temperature"]
        c["retrain"] = m.get("retrain", f"RM={m['name']} GPU=0 PORT=8100 bash scripts/remote/run_mcrm.sh")
        c["output"] = "单个字母 A / B / C 的 logprob；P(幻觉) = P(B) + P(C)，按 eval/report.json 的温度校准"
    elif meth == "sft":
        c.update({k: m[k] for k in ("config", "trainer", "data") if k in m})
        c["retrain"] = m.get("retrain", "")
        c["stats"] = {"train_rows": sum(1 for _ in open(root / m["data"][0], encoding="utf-8")) if (root / m["data"][0]).exists() else None}
    elif meth == "rag_grpo":
        g = dfl["rag_grpo"]
        c.update({"init": m["init"], "critic": m["critic"], "steps": m["steps"], "trainer": g["trainer"], "reward": g["reward"],
                  "prompts": g["prompts"], "eval": g["eval"], "script": g["script"], "retrain": m["retrain"]})
    elif meth == "gen_grpo":
        c.update({k: m[k] for k in ("init", "steps", "reward", "command") if k in m})
        c["retrain"] = m["command"]
    return c


def md_section(c: dict) -> str:
    L = [f"### {c['name']}", "", c["purpose"], ""]
    row = lambda k, v: L.append(f"- **{k}**：{v}")
    row("发布位置", f"`{c['release_path']}`")
    row("基座", c["base"])
    if c["used_in"]:
        row("对应结论", c["used_in"])
    if c["method"] == "option_rm":
        row("训练数据", "、".join(f"`{x}`" for x in c["data"]))
        if c.get("data_from"):
            row("数据怎么来的", c["data_from"])
        s = c.get("stats", {})
        if s.get("train"):
            lab = "，".join(f"{LABEL.get(k, k)} {v}" for k, v in sorted((s['train'].get('label') or {}).items()))
            row("训练集", f"{s['train']['n']} 行（{lab}）；最长 {s.get('max_tokens')} token")
        if s.get("build"):
            row("构建参数（meta.json）", "，".join(f"{k}={v}" for k, v in s["build"].items()))
        if s.get("temperature") is not None:
            row("校准温度", f"{s['temperature']}" + (f"（校准集原始 AUC {s['calib_auc']:.3f}）" if s.get("calib_auc") else ""))
        row("训练配置", f"`{c['config']}`；LoRA {c['lora']}；{c['trainer']}")
        row("输出", c["output"])
    elif c["method"] == "sft":
        row("训练数据", "、".join(f"`{x}`" for x in c["data"]) + (f"（{c['stats']['train_rows']} 行）" if c["stats"].get("train_rows") else ""))
        row("训练配置", f"`{c['config']}`；{c['trainer']}")
    elif c["method"] == "rag_grpo":
        row("起点", c["init"])
        row("奖励里的 critic", f"`{c['critic']}`")
        row("训练", f"{c['steps']} 步；{c['trainer']}")
        row("奖励", c["reward"])
        row("训练原文", c["prompts"])
        row("评测", c["eval"])
        row("训练日志", "`outputs/rt/core/loop2_<组>/run/log.jsonl`（逐步）、`quick_eval.jsonl`")
    elif c["method"] == "gen_grpo":
        row("起点", c["init"])
        row("训练", f"{c['steps']} 步")
        row("奖励", c["reward"])
    if c.get("retrain"):
        L += ["", "重训：", "", "```bash", c["retrain"], "```"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(ROOT / "configs/release_models.yaml"))
    ap.add_argument("--data_root", default=str(ROOT), help="解压了发布数据的目录（读 meta.json / report.json）")
    ap.add_argument("--out_md", default=str(ROOT / "docs/MODELS.md"))
    ap.add_argument("--cards_dir", default="", help="给了就为每个模型写 README.md + training.json")
    a = ap.parse_args()
    reg = yaml.safe_load(open(a.registry, encoding="utf-8"))
    cards = [card(m, reg["defaults"], Path(a.data_root)) for m in reg["models"]]

    md = ["# 模型清单：每个模型的训练数据和训练配置", "",
          "本页由 `python -m scripts.model_cards` 从 `configs/release_models.yaml` 和发布数据里的 `meta.json` / `eval/report.json` 生成，不要手改。",
          "发布包和复现方法见 [RELEASE.md](RELEASE.md)。所有模型都只发 LoRA。", "",
          "共同约定：",
          "- 打分模型（critic / sev / RM2 / 满意度 / 拒答）是\"单字母选项\"格式：模型只输出一个字母，取 logprob 当概率，再用校准集拟合一个全局温度；",
          "- 所有打分模型共用一份超参（`configs/rt/rm_mc.yaml`），区别只在训练数据；",
          "- 种子固定（seed 0、`PYTHONHASHSEED=0`）；RAG 和生成器的 GRPO 用 vLLM 采样，有少量非确定性。", "",
          "## 总表", "", "| 模型 | 类别 | 训练方式 | 起点 / 训练数据 | 对应结论 |", "|---|---|---|---|---|"]
    meth_name = {"option_rm": "选项 SFT", "sft": "SFT", "rag_grpo": "GRPO（R3 奖励）", "gen_grpo": "GRPO"}
    for c in cards:
        if c["method"] == "option_rm":
            n = (c.get("stats", {}).get("train") or {}).get("n")
            src = f"`outputs/rt/rm/{c['release_path'].split('/')[-1]}/` {n or '?'} 行"
        elif c["method"] == "sft":
            src = "、".join(f"`{x}`" for x in c["data"])
        elif c["method"] == "rag_grpo":
            src = f"起点 {c['init']}；critic `{c['critic']}`；{c['steps']} 步"
        else:
            src = f"起点 {c['init']}；{c['steps']} 步"
        md.append(f"| {c['name']} | {c['group']} | {meth_name[c['method']]} | {src} | {c['used_in']} |")
    for g, title in GROUP.items():
        md += ["", f"## {title}", ""]
        md += [md_section(c) for c in cards if c["group"] == g]
    Path(a.out_md).write_text("\n".join(md), encoding="utf-8", newline="\n")
    print("wrote", a.out_md, len(cards), "models")

    if a.cards_dir:
        for c in cards:
            d = Path(a.cards_dir) / c["release_path"]
            d.mkdir(parents=True, exist_ok=True)
            head = f"---\nlicense: other\nbase_model: {c['base'].split('@')[0]}\nlibrary_name: peft\ntags: [lora, claim-critic]\n---\n\n"
            body = md_section(c).replace(f"### {c['name']}", f"# {c['name']}", 1)
            foot = "\n项目说明、复现方法和许可见 https://github.com/liaomingyue/claim-critic（docs/MODELS.md、docs/RELEASE.md）。LoRA 权重遵循基座模型的许可证。\n"
            (d / "README.md").write_text(head + body + foot, encoding="utf-8", newline="\n")
            (d / "training.json").write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        print("wrote cards to", a.cards_dir)


if __name__ == "__main__":
    main()
