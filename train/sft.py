"""SFT。两种模式：

- coldstart：2000 条，think 用一句模板占位，只教格式和五类错误定义，不教推理 → critic-0。
- full：全量训练集，think 由 32B 模型根据真值写出（rationalize），作为「同基座全量 SFT」基线。

数据用 TRL 的 prompt-completion 格式，loss 只算在 completion 上。

用法：
  python -m train.sft --config configs/sft.yaml --mode coldstart
  python -m train.sft --config configs/sft.yaml --mode full
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from common import manifest
from common.config import load_config
from common.llm import LLM, LLMConfig, kept, map_resumable
from common.output_format import format_output, parse_evidence
from common.prompts import critic_messages, sample_input
from common.schema import read_jsonl

PLACEHOLDER_THINK = "先核对 claim 内容是否出自被引用的文献，再核对术语译名，最后逐项核对对象、条件和因果关系。"

RATIONALE_PROMPT = """下面是一条 claim、它的参考文献和已知的正确判定。请写出得出该判定的分析过程（3–6 句），
像审核员自己在思考一样，不要提到「已知答案」。只输出分析过程本身。

{input}

正确判定：{verdict}；错误类型：{type}；证据：{evidence}；原因：{reason}"""


def target_text(s: dict, think: str) -> str:
    ref_id, snippet = (None, None) if s["evidence"] == "none" else parse_evidence(s["evidence"])
    return format_output(s["label"], ref_id, snippet, s["type"], s.get("reason", ""), think=think)


def write_rationales(samples: list[dict], cfg: dict, out: str) -> dict[str, str]:
    llm = LLM(LLMConfig.from_dict(cfg["rationale_model"]))

    def run(s: dict) -> dict:
        p = RATIONALE_PROMPT.format(input=sample_input(s), verdict=s["label"],
                                    type=s["type"], evidence=s["evidence"], reason=s.get("reason", ""))
        return {"think": llm.chat_one([{"role": "user", "content": p}], temperature=0.3).strip()}

    map_resumable(samples, run, out, concurrency=llm.cfg.concurrency)
    return {r["id"]: r["think"] for r in kept(out)}


def build_dataset(samples: list[dict], thinks: dict[str, str] | None, task: str = "claim", tips: bool = False,
                  style: str = "full"):
    """task: claim（医学综述 claim，默认）| ragtruth（逐句，见 ragtruth.task）| ragtruth_resp（回答级，见 ragtruth.resp）
    | discrim（真实性判别器，见 ragtruth.discrim）| gen（错误生成器，见 ragtruth.realgen）| option（单字符选项 RM，见 ragtruth.rm_data）。
    style：ragtruth 为 full | verdict；ragtruth_resp 为 plain | evidence。"""
    from datasets import Dataset

    rows = []
    for s in samples:
        if task == "ragtruth":  # 逐句样本（ragtruth.build 产出），evidence / reason 已由 ragtruth.annotate 写好
            from ragtruth import task as rt
            rows.append({"prompt": rt.messages(s, tips, style),
                         "completion": [{"role": "assistant", "content": rt.target_text(s, style)}]})
            continue
        if task == "discrim":  # 真实性判别器：自然错误 vs 合成错误（ragtruth.discrim）
            from ragtruth import discrim as dm
            rows.append({"prompt": dm.messages(s), "completion": [{"role": "assistant", "content": dm.target_text(s)}]})
            continue
        if task == "option":  # 单字符选项 RM（ragtruth.rm_data）：prompt 是 system + user 消息，completion 是一个选项字母
            rows.append({"prompt": s["prompt"], "completion": [{"role": "assistant", "content": s["completion"]}]})
            continue
        if task == "policy_text":  # 回答策略 SFT（2026-11-05，S1 / S2）：prompt 是 GRPO 采样时同一段聊天模板文本（ans_data.policy_prompt，关思考），completion 是回答 + <|im_end|>
            rows.append({"prompt": s["prompt"], "completion": s["completion"]})
            continue
        if task == "gen":  # 错误生成器：prompt / completion 已由 ragtruth.realgen build 拼好（生成器 prompt + 输出 JSON）
            rows.append({"prompt": [{"role": "user", "content": s["prompt"]}],
                         "completion": [{"role": "assistant", "content": s["completion"]}]})
            continue
        if task == "ragtruth_resp":  # 回答样本；evidence 风格需要 analysis
            from ragtruth import resp as rr
            rows.append({"prompt": rr.messages(s, style),
                         "completion": [{"role": "assistant", "content": rr.target_text(s, style)}]})
            continue
        think = thinks.get(s["id"], PLACEHOLDER_THINK) if thinks else PLACEHOLDER_THINK
        rows.append({
            "prompt": critic_messages(s),
            "completion": [{"role": "assistant", "content": target_text(s, think)}],
        })
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--mode", choices=["coldstart", "full"], required=True)
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_config(args.config)
    mcfg = cfg[args.mode]

    samples = list(read_jsonl(mcfg["data"]))
    thinks = write_rationales(samples, cfg, mcfg["rationale_cache"]) if args.mode == "full" else None
    dataset = build_dataset(samples, thinks, cfg.get("task", "claim"), cfg.get("prompt_tips", False),
                            cfg.get("style", "full"))

    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    targs = SFTConfig(output_dir=mcfg["output_dir"], **cfg["trainer"], **mcfg.get("trainer", {}))
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    peft_config = None
    if cfg.get("use_lora"):
        lora = dict(cfg["lora"])
        if cfg.get("lora_train_tokens"):  # Base 模型的对话特殊 token 词向量没训练过，LoRA 碰不到词向量，单独放开这几行
            lora["trainable_token_indices"] = {"embed_tokens": tok.convert_tokens_to_ids(cfg["lora_train_tokens"])}
        peft_config = LoraConfig(**lora)
    trainer = SFTTrainer(model=cfg["base_model"], args=targs, train_dataset=dataset,
                         processing_class=tok, peft_config=peft_config)
    trainer.train()
    if peft_config:   # LoRA 适配器单独存一份（约 0.2 GB）：跨机器拷贝、归档用；合并后的完整权重给 vLLM / GRPO 直接加载
        trainer.model.save_pretrained(f"{mcfg['output_dir']}/adapter")
    model = trainer.model.merge_and_unload() if peft_config and cfg.get("merge_lora") else trainer.model
    if cfg.get("eos_tokens"):  # Base 模型默认只在 <|endoftext|> 停，推理时（transformers / vLLM）要在回合结束符处停
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        model.generation_config.eos_token_id = tok.convert_tokens_to_ids(cfg["eos_tokens"]) + eos
    if model is not trainer.model:  # 合并成完整权重，后续 GRPO / vLLM 直接加载
        model.save_pretrained(mcfg["output_dir"])
        tok.save_pretrained(mcfg["output_dir"])
    else:
        trainer.save_model(mcfg["output_dir"])
    with open(f"{mcfg['output_dir']}/sft_meta.json", "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "n_samples": len(samples), "base_model": cfg["base_model"]}, f)
    Path(mcfg["output_dir"], "SAVED").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log = trainer.state.log_history

    def _manifest():
        manifest.write("run", Path(mcfg["output_dir"]).name, mcfg["output_dir"], {"config_file": args.config, **cfg},
                       [mcfg["data"], args.config], started=t0,
                       extra={"mode": args.mode, "n_samples": len(samples), "base_model": cfg["base_model"],
                              "adapter_dir": f"{mcfg['output_dir']}/adapter" if peft_config else None,
                              "train": {"steps": trainer.state.global_step,
                                        "runtime_s": next((x["train_runtime"] for x in log if "train_runtime" in x), None),
                                        "final_loss": next((x["loss"] for x in reversed(log) if "loss" in x), None)}})
    # 2026-11-05：大数据量的 critic（z6000 / z3000）存完模型后进程非零退出或卡住（原因未查明，疑在收尾的清单 / 共享盘写入），
    # 卡住时 GPU 空闲被看门狗关机。模型存完先写 SAVED；清单放后台线程最多等 120 秒、出错只打印；最后直接退出进程
    import threading
    err = []
    th = threading.Thread(target=lambda: _run_safe(_manifest, err), daemon=True)
    th.start()
    th.join(120)
    if th.is_alive() or err:
        print("SFT_MANIFEST_SKIPPED", "timeout" if th.is_alive() else err[0], flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _run_safe(fn, err: list) -> None:
    import traceback
    try:
        fn()
    except Exception:  # noqa: BLE001 清单只是记录，失败不影响模型
        err.append(traceback.format_exc()[-500:])


if __name__ == "__main__":
    main()
