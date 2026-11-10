"""去掉 LoRA 适配器里视觉塔的参数：TRL 按多模态结构加载 Qwen3.5，all-linear 把 visual.* 也挂了 LoRA，
vLLM 多 LoRA 服务只接受语言模型的模块，加载报错。纯文本打分用不到视觉塔，删掉不影响结果。

  python scripts/strip_visual_lora.py <适配器目录> <输出目录>
"""
import json
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
w = load_file(str(src / "adapter_model.safetensors"))
keep = {k: v for k, v in w.items() if ".visual." not in k and not k.startswith("visual.")}
save_file(keep, str(dst / "adapter_model.safetensors"))
cfg = json.loads((src / "adapter_config.json").read_text())
mods = sorted({k.split(".lora_")[0].split(".")[-1] for k in keep if ".lora_" in k})
cfg["target_modules"] = mods
(dst / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
for f in src.iterdir():
    if f.name not in ("adapter_model.safetensors", "adapter_config.json") and f.is_file():
        shutil.copy(f, dst / f.name)
print(f"{src}: {len(w)} tensors -> {len(keep)} (dropped {len(w) - len(keep)} visual), target_modules {mods}")
