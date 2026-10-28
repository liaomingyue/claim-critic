"""从已保存的 LoRA 适配器合并出完整权重（与 train.sft 结尾的保存逻辑一致：合并 → 设回合结束符 → 存权重和分词器）。

用途：训练完成但合并保存失败（如磁盘满）时补救，不用重训；或把只留了适配器的模型恢复成 vLLM 可直接加载的完整权重。
用法：python -m train.merge_lora --config configs/rt/exp/discrim_9b_r2.yaml [--mode coldstart] [--log 训练日志]
      python -m train.merge_lora --ckpt /root/autodl-tmp/ckpt/critic-gan-r2 --base /root/autodl-tmp/models/Qwen3.5-9B
      （第二种不需要训练配置：新机器从文件存储拷来适配器后直接合并；适配器目录自带的 manifest 保留不动）
"""
from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

from common import manifest
from common.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="训练配置（与 --ckpt 二选一）")
    ap.add_argument("--mode", default="coldstart")
    ap.add_argument("--log", help="原训练日志，用来补 manifest 里的步数 / 耗时 / loss")
    ap.add_argument("--ckpt", help="含 adapter/ 的目录，合并后的权重写在同一目录")
    ap.add_argument("--base", default="/root/autodl-tmp/models/Qwen3.5-9B")
    ap.add_argument("--eos_tokens", nargs="*", default=["<|im_end|>"])
    args = ap.parse_args()
    t0 = time.time()
    if args.ckpt:
        cfg, mcfg, out = {"base_model": args.base, "eos_tokens": args.eos_tokens}, None, args.ckpt
    else:
        cfg = load_config(args.config)
        mcfg = cfg[args.mode]
        out = mcfg["output_dir"]

    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    # 训练时 TRL 按哪种结构加载，适配器参数名就带哪种前缀：多模态结构是 model.language_model.layers…，纯文本是 model.layers…。
    # 结构选错时 PEFT 只给一条警告、一个参数都不合并，得到的就是基座（踩过：D2 第一次合并）
    with safe_open(f"{out}/adapter/adapter_model.safetensors", "pt") as f:
        keys = list(f.keys())
    cls = AutoModelForImageTextToText if any(".language_model." in k for k in keys) else AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = cls.from_pretrained(cfg["base_model"], dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, f"{out}/adapter")
    lora_b = {n: p for n, p in model.named_parameters() if "lora_B" in n}
    # lora_B 初始化为 0，没加载上的仍是 0；视觉塔上的 LoRA 在纯文本训练里没有梯度，本来就是 0，不算
    text_b = [p for n, p in lora_b.items() if ".visual." not in n]
    loaded = sum(int(p.abs().sum().item() > 0) for p in text_b)
    if len(lora_b) * 2 != len(keys) or loaded < len(text_b) * 0.95:
        raise SystemExit(f"adapter not applied: {len(keys)} keys, {len(text_b)} text lora_B modules, {loaded} non-zero")
    print(f"adapter applied via {cls.__name__}: {loaded}/{len(text_b)} text lora_B non-zero", flush=True)
    model = model.merge_and_unload()
    if cfg.get("eos_tokens"):
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        model.generation_config.eos_token_id = tok.convert_tokens_to_ids(cfg["eos_tokens"]) + eos
    model.save_pretrained(out)
    tok.save_pretrained(out)

    if mcfg is None:   # 只合并：训练记录（run_manifest.json）随适配器一起保存过，不重写
        return
    train = {}
    if args.log:   # 训练日志里 trainer 打印的 dict 行
        for line in Path(args.log).read_text(encoding="utf-8", errors="ignore").replace("\r", "\n").splitlines():
            if line.startswith("{'"):
                d = ast.literal_eval(line)
                if "train_runtime" in d:
                    train["runtime_s"] = float(d["train_runtime"])
                elif "loss" in d:
                    train["final_loss"] = float(d["loss"])
    n = sum(1 for _ in open(mcfg["data"], encoding="utf-8"))
    with open(f"{out}/sft_meta.json", "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "n_samples": n, "base_model": cfg["base_model"]}, f)
    manifest.write("run", Path(out).name, out, {"config_file": args.config, **cfg}, [mcfg["data"], args.config],
                   started=t0, extra={"mode": args.mode, "n_samples": n, "base_model": cfg["base_model"],
                                      "adapter_dir": f"{out}/adapter", "train": train,
                                      "note": "merged from saved adapter by train.merge_lora (original save failed)"})


if __name__ == "__main__":
    main()
