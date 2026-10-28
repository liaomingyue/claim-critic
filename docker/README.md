# Docker：在 Windows 上跑 vLLM 和 verl

vLLM 和 verl 只支持 Linux。Windows 上有两种方式运行这里的两个容器：

- **WSL2 里的 Docker Engine（不需要管理员权限）**：在 WSL 的 Ubuntu 里装 `docker-ce` 和 `nvidia-container-toolkit`，然后 `nvidia-ctk runtime configure --runtime=docker && systemctl restart docker`。命令都在 WSL 里、仓库的 `/mnt/<盘符>/...` 路径下运行，从 Windows 调用写成 `wsl -d Ubuntu-24.04 -u root -- bash -lc "cd /mnt/i/code/critic_model && docker compose ..."`。
- **Docker Desktop**（安装需要管理员确认）：设置里勾选 Use the WSL 2 based engine，命令直接在 PowerShell 里运行。

| 服务 | 镜像 | 用途 |
|---|---|---|
| `vllm` | `vllm/vllm-openai:v0.31.0` | OpenAI 兼容推理服务，供 `eval.run_eval`、`data.mine_hard`、`infer.judge`、`app.main` 调用 |
| `verl` | `verlai/verl:vllm024.dev2` + `verl==0.9.1`（`verl.Dockerfile`） | 跑 `train.grpo launch` |

仓库根目录挂载到容器的 `/work`，HuggingFace 缓存挂载到 `/root/.cache/huggingface`，所以模型只下载一次，产出直接落在本机的 `outputs/`。以下命令都在仓库根目录运行。

## 0. 确认容器能用 GPU

```powershell
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi
```

能看到显卡就行，不用在 WSL 里另装驱动。

## 1. vLLM 服务

```powershell
docker compose -f docker/compose.yaml up -d vllm           # 默认加载 outputs/smoke/ckpt/critic-0，端口 8002
docker compose -f docker/compose.yaml logs -f vllm          # 看到 "Application startup complete" 即可

# 换模型（PowerShell 写法；bash 里写成 VLLM_MODEL=... docker compose ...）
$env:VLLM_MODEL="outputs/ckpt/critic-3"; $env:VLLM_NAME="critic-3"; docker compose -f docker/compose.yaml up -d vllm
```

可调的环境变量：`VLLM_MODEL`（相对仓库根目录的模型目录）、`VLLM_NAME`、`VLLM_PORT`（默认 8002）、`VLLM_MAX_LEN`（默认 4096）、`VLLM_GPU_MEM`（默认 0.85）、`HF_ENDPOINT`（国内用 `https://hf-mirror.com`）、`HF_CACHE`（WSL 里建议指向 Windows 侧同一个缓存，如 `/mnt/i/cache/huggingface`，模型不用下两遍）。

调用方在 Windows 本机跑即可，只需要 `openai` 包：

```powershell
python -m eval.run_eval --test outputs/smoke/trainset_v1/val.jsonl --model critic-0 --base_url http://localhost:8002/v1 --out outputs/smoke/eval_vllm
```

12 GB 卡上：0.6B / 1.7B 宽裕，4B（bf16 约 8 GB）把 `VLLM_MAX_LEN` 压到 4096 左右能跑，8B 需要 AWQ 等量化版本。

### WSL 里跑的已知问题（RTX 4070 SUPER 12 GB 上实测）

- **WSL 空闲约 1 分钟会自动关机**，里面的 Docker 和 vLLM 一起停（容器状态 `Exited (255)`）。服务要常驻时另开一个会话挂着：`wsl -d Ubuntu-24.04 -u root -- sleep infinity`。
- **WSL 下锁页内存不可用**（日志：`Pinned memory is not available on this platform`），缓冲区改占显存，显卡显示还有几 GB 空闲也会报 OOM。12 GB 卡用 `VLLM_GPU_MEM=0.4 VLLM_EXTRA=--enforce-eager` 跑 0.6B 正常。
- **拉大镜像后 WSL 的页缓存会占住 Windows 内存**（任务管理器里 `vmmemWSL` 十几 GB），Windows 侧内存告急时 CUDA 分配也会失败。拉完镜像执行 `wsl -u root -- bash -c "sync; echo 3 > /proc/sys/vm/drop_caches"` 释放；长期可在 `.wslconfig` 里设 `memory=` 上限。
- WSL 会把环境变量 `NAME` 设成主机名，所以 compose 的变量都带 `VLLM_` 前缀。
- Windows 侧访问用 `http://127.0.0.1:8002/v1`。

## 2. verl 训练

```powershell
docker compose -f docker/compose.yaml build verl            # 首次构建，基础镜像较大
docker compose -f docker/compose.yaml run --rm verl python -m train.grpo prepare --config configs/smoke/grpo.yaml --version 1
docker compose -f docker/compose.yaml run --rm verl python -m train.grpo launch  --config configs/smoke/grpo.yaml --version 1
```

`configs/smoke/grpo.yaml` 是单卡 0.6B、5 步的配置（vLLM 占 30% 显存，FSDP 参数和优化器卸载到 CPU），只验证 verl 的参数名、reward 函数和数据格式能跑通。跑之前先 `docker compose stop vllm` 释放显存。正式的 8B 训练用 `configs/grpo.yaml`，在多卡 Linux 机器上跑同一个镜像即可。

WSL 下还要几处额外设置：

- `-e VLLM_USE_V2_MODEL_RUNNER=0`：vLLM 0.24 的 V2 model runner 强制要求锁页内存，WSL 没有，报 `UVA is not available`；V1 会自动关掉 pin_memory。
- `-e PYTHONPATH=/work:/work/docker/wsl`：加载 `docker/wsl/sitecustomize.py`。WSL2 不支持 CUDA IPC，verl 却对 CUDA 写死支持，actor → vLLM 同步权重时报 `CUDA error: out of memory`（实为 IPC 失败）；补丁让 verl 改走共享内存传权重，只在 WSL 内核上生效。
- 单卡 + LoRA 还要关掉 `rollout.layered_summon`（已写在 `configs/smoke/grpo.yaml`），否则报 `offload_to_cpu=True and NO_SHARD is not supported yet`。


```powershell
docker compose -f docker/compose.yaml run --rm -e VLLM_USE_V2_MODEL_RUNNER=0 -e PYTHONPATH=/work:/work/docker/wsl verl python -m train.grpo launch --config configs/smoke/grpo.yaml --version 1
```

**12 GB 显存 / 32 GB 内存的 Windows 机器上实测跑通 verl GRPO 5 步**（2026-10-23，0.6B + LoRA，每步约 20–55 秒）。内存是唯一的瓶颈：verl 0.9 由 Ray 拉起约 10 个 Python 进程，共享内存传权重还要再占几 GB；WSL 里 CUDA 分配显存又要占 Windows 的内存提交额度。需要同时满足：

- `.wslconfig` 里 `memory=18GB`（14 GB 在权重同步时被 Ray 杀进程）；
- Windows 页面文件放大（放在数据盘，如 I 盘 4–16 GB，系统可自动增长），否则 WSL 一进入训练就被 Windows 整体关掉；
- 冒烟配置里的省内存设置：LoRA、agent loop / reward / dataloader 各 1 个进程、Ray 对象存储 1 GB、权重传输桶 512 MB。

做不到时，本机用 TRL 路径（`train.grpo_trl`）验证 GRPO，verl 只在 Linux 训练机上跑。

verl 的 hydra 参数名随版本变化；`launch` 报 `Could not override` 或 `Key ... not in struct` 时，按报错改 `train/grpo.py:verl_overrides` 或配置里的 `extra_overrides`。

## 注意

- 挂载 Windows 盘读写较慢，冒烟测试无所谓；正式数据量大时把仓库放到 WSL 文件系统（`\\wsl$\Ubuntu-24.04\home\...`）里再挂载。
- 升级镜像时改 `compose.yaml` / `verl.Dockerfile` 里的版本号，然后重跑 `python -m scripts.env_check`（容器里：`docker compose -f docker/compose.yaml run --rm verl python -m scripts.env_check`）。
