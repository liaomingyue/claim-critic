"""错误合成模型（生成器）的 DPO 训练：TRL DPOTrainer + LoRA。偏好对来自 ragtruth.gen_dpo pairs。

参考模型 = 关掉 LoRA 的同一个基座（TRL 对 PEFT 模型的默认做法），不另外加载一份，9B 在 32 GB 卡上放得下。
产出：<output_dir>/adapter（LoRA 适配器）+ 合并后的完整权重（vLLM 直接加载）+ run_manifest.json。

用法：python -m train.dpo --config configs/rt/dpo_gen_9b.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import manifest
from common.config import load_config
from common.schema import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_config(args.config)

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    rows = [{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in read_jsonl(cfg["data"])]
    dataset = Dataset.from_list(rows)
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    targs = DPOConfig(output_dir=cfg["output_dir"], **cfg["trainer"])
    trainer = DPOTrainer(model=cfg["base_model"], args=targs, train_dataset=dataset, processing_class=tok,
                         peft_config=LoraConfig(**cfg["lora"]))
    trainer.train()
    out = cfg["output_dir"]
    trainer.model.save_pretrained(f"{out}/adapter")
    model = trainer.model.merge_and_unload()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    log = trainer.state.log_history
    last = next((x for x in reversed(log) if "loss" in x), {})
    manifest.write("run", Path(out).name, out, {"config_file": args.config, **cfg}, [cfg["data"], args.config],
                   started=t0, extra={"base_model": cfg["base_model"], "adapter_dir": f"{out}/adapter",
                                      "train": {"steps": trainer.state.global_step, "final_loss": last.get("loss"),
                                                "rewards_accuracy": last.get("rewards/accuracies"),
                                                "rewards_margin": last.get("rewards/margins")}})


if __name__ == "__main__":
    main()
