"""WSL2 下给 verl 打的运行时补丁，只在 WSL 内核上生效。

WSL2 不支持 CUDA IPC，而 verl 的 CUDA 平台写死 is_ipc_supported() = True，
actor → vLLM 同步权重时 _share_cuda_ 报 "CUDA error: out of memory"。
改成 False 后 verl 走共享内存（/dev/shm）传权重，这条路径本来是给 NPU 用的。

用法（只在 WSL 里）：docker compose run -e PYTHONPATH=/work:/work/docker/wsl ...
Python 启动时自动 import sitecustomize；Ray 的子进程继承 PYTHONPATH，也会生效。
只挂一个 import 钩子，等进程真的导入 verl 的 CUDA 平台模块时才改；不在这里 import verl，
否则每个 Python 进程（含 Ray dashboard 等）都会顺带加载 torch，内存翻几倍。
"""
import importlib.abc
import importlib.machinery
import platform
import sys

TARGET = "verl.plugin.platform.platform_cuda"


class _PatchAfterLoad(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def patched_exec(module):
            exec_module(module)
            module.PlatformCUDA.is_ipc_supported = lambda self: False

        spec.loader.exec_module = patched_exec
        return spec


if "microsoft" in platform.release().lower():
    sys.meta_path.insert(0, _PatchAfterLoad())
