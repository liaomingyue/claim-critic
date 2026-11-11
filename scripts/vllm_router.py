"""推理机上的转发层：一个对外端口后面挂两个本机 vLLM（不同底座），按请求里的 model 名分流。

9B 策略 × 2B 打分模型的多机 GRPO 要在一张推理卡上同时起 9B（策略采样）和 2B（critic / 满意度 / 拒答）两个服务，
而平台对外只开 6006、6008 两个端口（6008 给 LoRA 同步），所以 6006 放这个转发层，训练机那边不用改：
  - POST（completions / chat completions 等）：body 里的 model 在 --rm_models 里 → 打分服务，其余（policy_v0、各版本策略 LoRA）→ 策略服务
  - GET /v1/models：两边的模型列表合并；GET /metrics：两边原文拼接（ans_grpo.cache_hits 按正则加总）
  - 其它路径（load_lora_adapter / unload_lora_adapter 等）→ 策略服务
所有请求要带和 vLLM 相同的 API 密钥，转发时原样带上。

  VLLM_API_KEY=... python scripts/vllm_router.py --port 6006 --policy http://127.0.0.1:8000 --rm http://127.0.0.1:8001
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request as urlreq

HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "host"}


def forward(base: str, method: str, path: str, body: bytes | None, headers: dict) -> tuple[int, dict, bytes]:
    h = {k: v for k, v in headers.items() if k.lower() not in HOP}
    req = urlreq.Request(base.rstrip("/") + path, data=body, method=method, headers=h)
    try:
        with urlreq.urlopen(req, timeout=1800) as r:
            return r.status, dict(r.headers), r.read()
    except error.HTTPError as e:
        return e.code, dict(e.headers), e.read()
    except Exception as e:  # noqa: BLE001  上游没起来 / 断开
        return 502, {"Content-Type": "application/json"}, json.dumps({"error": f"upstream {base}: {e}"}).encode()


def make_handler(policy: str, rm: str, key: str, rm_models: set[str]):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"   # 客户端（openai / httpx）用长连接；每个回复都带 Content-Length

        def log_message(self, *a) -> None:
            pass

        def _send(self, code: int, headers: dict, body: bytes) -> None:
            self.send_response(code)
            for k, v in headers.items():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authed(self) -> bool:
            return hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {key}")

        def do_GET(self) -> None:
            if self.path == "/health":
                ok = all(forward(u, "GET", "/health", None, {})[0] == 200 for u in (policy, rm))
                return self._send(200 if ok else 503, {"Content-Type": "application/json"}, json.dumps({"ok": ok}).encode())
            if not self._authed():
                return self._send(401, {"Content-Type": "application/json"}, b'{"error":"bad key"}')
            hs = dict(self.headers)
            if self.path.startswith("/v1/models"):
                rs = [forward(u, "GET", self.path, None, hs) for u in (policy, rm)]
                bad = next((r for r in rs if r[0] != 200), None)
                if bad:
                    return self._send(*bad)
                docs = [json.loads(r[2]) for r in rs]
                merged = {**docs[0], "data": docs[0].get("data", []) + docs[1].get("data", [])}
                return self._send(200, {"Content-Type": "application/json"}, json.dumps(merged).encode())
            if self.path.startswith("/metrics"):
                rs = [forward(u, "GET", self.path, None, hs) for u in (policy, rm)]
                return self._send(200, {"Content-Type": "text/plain"}, b"\n".join(r[2] for r in rs if r[0] == 200))
            return self._send(*forward(policy, "GET", self.path, None, hs))

        def do_POST(self) -> None:
            if not self._authed():
                return self._send(401, {"Content-Type": "application/json"}, b'{"error":"bad key"}')
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            model = ""
            try:
                model = str(json.loads(body or b"{}").get("model", ""))
            except Exception:  # noqa: BLE001  不是 JSON 就按策略服务转
                pass
            up = rm if model in rm_models else policy
            return self._send(*forward(up, "POST", self.path, body, dict(self.headers)))

    return H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6006)
    ap.add_argument("--policy", default="http://127.0.0.1:8000", help="策略服务（不带 /v1）")
    ap.add_argument("--rm", default="http://127.0.0.1:8001", help="打分服务（不带 /v1）")
    ap.add_argument("--rm_models", default="critic,sat,refuse")
    a = ap.parse_args()
    key = os.environ.get("VLLM_API_KEY", "")
    assert key and key != "x", "VLLM_API_KEY 必须设（对外域名访问）"
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(a.policy, a.rm, key, set(a.rm_models.split(","))))
    srv.daemon_threads = True
    print(f"VLLM_ROUTER up on {a.port} (policy {a.policy}, rm {a.rm} for {a.rm_models})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
