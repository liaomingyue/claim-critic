"""把一个 LoRA 的 B 矩阵清零，得到「等于基座」的起点 LoRA（结构和原来的一样，训练和 vLLM 热加载都能直接用）。
GRPO 从「基座 + 提示词」起步时用：不经过监督微调（种子错误对微调会把模型听指令的能力训掉）。
  python scripts/zero_lora.py <源 adapter 目录> <输出目录>
"""
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.is_file() and f.name != "adapter_model.safetensors":
        shutil.copy(f, dst / f.name)
t = load_file(str(src / "adapter_model.safetensors"))
n = 0
for k in t:
    if "lora_B" in k:
        t[k] = t[k] * 0
        n += 1
save_file(t, str(dst / "adapter_model.safetensors"))
print("ZERO_LORA", len(t), "tensors,", n, "lora_B zeroed ->", dst)
