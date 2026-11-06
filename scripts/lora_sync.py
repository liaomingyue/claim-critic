"""双机 GRPO 的 LoRA 同步服务，跑在推理机上（2026-11-04，用户：抢不到双卡就两台单卡，一台训练一台推理；
每步训完存到共享盘，推理机直接加载到内存）。

两种推送方式（训练机 train.gen_grpo.Server 按 LORA_SYNC_MODE 选，默认 upload）：
  PUT /push/<name>：请求体是 LoRA 目录的 tar（不压缩），直接写到推理机内存盘 /dev/shm/lora_sync/<name> 再热加载——不经共享盘
                    （用户：不写共享盘，直接传给推理机；实测 100 MB 上传 1.5 秒、网关 200 MB 以内不限大小）
  POST /sync {"name", "src"}：训练机先存到共享盘，这里把 src 目录拷到内存盘再热加载（备用）
POST /unload {"name"} 卸载并删掉内存盘副本。所有接口都要带和 vLLM 相同的 API 密钥（经 AutoDL 对外域名访问）。
训练机那边只要设 LORA_SYNC_URL=https://<推理机 6008 域名>，train.gen_grpo.Server 的 load_lora / unload_lora 就会改走这里。

  VLLM_API_KEY=... python scripts/lora_sync.py --port 6008 --vllm http://127.0.0.1:6006/v1
实测（2026-11-04，两台无卡实例）：共享盘写 100 MB 1.2–2.1 秒（训练机），读出并拷到内存盘 0.4 秒（推理机）。
"""
from __future__ import annotations

import argparse
import hmac
import io
import json
import os
import shutil
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import request as urlreq


def load(vllm: str, key: str, name: str, path: str) -> tuple[int, str]:
    """本机 vLLM 热加载；同名已加载（回退 / 续训）就先卸载再加载。"""
    def post(ep: str, body: dict) -> tuple[int, str]:
        req = urlreq.Request(f"{vllm}/{ep}", data=json.dumps(body).encode(), method="POST",
                             headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        try:
            with urlreq.urlopen(req, timeout=600) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urlreq.HTTPError as e:   # noqa: PERF203
            return e.code, e.read().decode("utf-8", "replace")

    code, text = post("load_lora_adapter", {"lora_name": name, "lora_path": path})
    if code == 400 and "already been loaded" in text:
        post("unload_lora_adapter", {"lora_name": name})
        code, text = post("load_lora_adapter", {"lora_name": name, "lora_path": path})
    return code, text


def safe_extract(data: bytes, dst: Path) -> None:
    """只接受普通文件和目录、相对路径、不含 ..（请求来自对外域名，虽然有密钥也不信任 tar 里的路径）。"""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
        ms = t.getmembers()
        for m in ms:
            if not (m.isfile() or m.isdir()) or m.name.startswith(("/", "\\")) or ".." in Path(m.name).parts:
                raise ValueError(f"bad tar member {m.name}")
        dst.mkdir(parents=True, exist_ok=True)
        for m in ms:
            out = dst / m.name
            if m.isdir():
                out.mkdir(parents=True, exist_ok=True)
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(t.extractfile(m).read())


def make_handler(vllm: str, key: str, shm: Path, dry: bool):
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def _reply(self, code: int, obj: dict) -> None:
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *a) -> None:   # 不刷屏
            pass

        def do_GET(self) -> None:
            self._reply(200, {"ok": True}) if self.path == "/health" else self._reply(404, {})

        def _authed(self) -> bool:
            return hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {key}")

        def _load_reply(self, name: str, dst: Path, t0: float, t1: float) -> None:
            code, text = (200, "dry run") if dry else load(vllm, key, name, str(dst))
            t2 = time.time()
            if code != 200:
                return self._reply(502, {"error": f"vllm load {code}: {text[:300]}"})
            mb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 2**20
            return self._reply(200, {"name": name, "mb": round(mb, 1), "t_copy": round(t1 - t0, 3), "t_load": round(t2 - t1, 3)})

        def do_PUT(self) -> None:
            if not self._authed():
                return self._reply(401, {"error": "bad key"})
            name = self.path.removeprefix("/push/")
            if not self.path.startswith("/push/") or not name or "/" in name or ".." in name:
                return self._reply(400, {"error": "bad path"})
            n = int(self.headers.get("Content-Length", 0))
            t0 = time.time()
            data = self.rfile.read(n)
            dst = shm / name
            with lock:
                try:
                    shutil.rmtree(dst, ignore_errors=True)
                    safe_extract(data, dst)
                except Exception as e:  # noqa: BLE001
                    shutil.rmtree(dst, ignore_errors=True)
                    return self._reply(400, {"error": f"bad upload: {e}"})
                return self._load_reply(name, dst, t0, time.time())

        def do_POST(self) -> None:
            if not self._authed():
                return self._reply(401, {"error": "bad key"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                name = str(body["name"])
                assert name and "/" not in name and ".." not in name
            except Exception as e:  # noqa: BLE001
                return self._reply(400, {"error": f"bad request: {e}"})
            dst = shm / name
            with lock:   # 同一时刻只同步一个版本，避免拷贝和加载交错
                if self.path == "/sync":
                    src = Path(str(body.get("src", "")))
                    if not src.is_dir():
                        return self._reply(400, {"error": f"src not a dir: {src}"})
                    t0 = time.time()
                    shutil.rmtree(dst, ignore_errors=True)
                    shutil.copytree(src, dst)
                    return self._load_reply(name, dst, t0, time.time())
                if self.path == "/unload":
                    if not dry:
                        req = urlreq.Request(f"{vllm}/unload_lora_adapter", data=json.dumps({"lora_name": name}).encode(), method="POST",
                                             headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
                        try:
                            urlreq.urlopen(req, timeout=600).read()
                        except Exception:  # noqa: BLE001  已经卸载过也没关系
                            pass
                    shutil.rmtree(dst, ignore_errors=True)
                    return self._reply(200, {"name": name, "unloaded": True})
            return self._reply(404, {})

    return H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6008)
    ap.add_argument("--vllm", default="http://127.0.0.1:6006/v1")
    ap.add_argument("--shm", default="/dev/shm/lora_sync")
    ap.add_argument("--dry", action="store_true", help="只拷贝不调用 vLLM（无卡机上验证共享盘 → 内存盘的速度和接口）")
    a = ap.parse_args()
    key = os.environ.get("VLLM_API_KEY", "")
    assert key and key != "x", "VLLM_API_KEY 必须设（对外域名访问）"
    shm = Path(a.shm)
    shm.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(a.vllm, key, shm, a.dry))
    print(f"LORA_SYNC up on {a.port} (vllm {a.vllm}, shm {shm}, dry {a.dry})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
