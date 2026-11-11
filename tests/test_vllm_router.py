"""推理机转发层：按 model 名分流到两个上游、合并模型列表和 metrics、密钥校验、上游报错原样回传。"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request

import pytest

from scripts import vllm_router


def _fake_upstream(tag: str):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self, code, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/health":
                return self._reply(200, b"")
            if self.headers.get("Authorization") != "Bearer k3y":
                return self._reply(401, {"error": "upstream bad key"})
            if self.path == "/v1/models":
                return self._reply(200, {"object": "list", "data": [{"id": f"{tag}_m"}]})
            if self.path == "/metrics":
                return self._reply(200, f"vllm:prefix_cache_hits_total {1 if tag == 'pol' else 2}.0".encode(), "text/plain")
            return self._reply(404, {})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if body.get("model") == "boom":
                return self._reply(400, {"error": "bad model"})
            return self._reply(200, {"upstream": tag, "path": self.path, "model": body.get("model")})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture()
def router():
    pol, rm = _fake_upstream("pol"), _fake_upstream("rm")
    h = vllm_router.make_handler(f"http://127.0.0.1:{pol.server_address[1]}", f"http://127.0.0.1:{rm.server_address[1]}", "k3y", {"critic", "sat", "refuse"})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    for s in (srv, pol, rm):
        s.shutdown()


def _req(url, body=None, key="k3y"):
    h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    data = json.dumps(body).encode() if body is not None else None
    try:
        with request.urlopen(request.Request(url, data=data, headers=h, method="POST" if data else "GET"), timeout=10) as r:
            return r.status, r.read()
    except error.HTTPError as e:
        return e.code, e.read()


def test_routes_by_model(router):
    for m, up in [("critic", "rm"), ("sat", "rm"), ("refuse", "rm"), ("policy_v0", "pol"), ("r3f_12", "pol")]:
        code, b = _req(router + "/v1/completions", {"model": m, "prompt": "x"})
        assert code == 200 and json.loads(b) == {"upstream": up, "path": "/v1/completions", "model": m}
    code, b = _req(router + "/v1/chat/completions", {"model": "critic", "messages": []})
    assert json.loads(b)["upstream"] == "rm"
    code, b = _req(router + "/v1/load_lora_adapter", {"lora_name": "r3f_3", "lora_path": "/x"})
    assert json.loads(b)["upstream"] == "pol"


def test_models_and_metrics_merged(router):
    code, b = _req(router + "/v1/models")
    assert code == 200 and [d["id"] for d in json.loads(b)["data"]] == ["pol_m", "rm_m"]
    code, b = _req(router + "/metrics")
    import re
    assert sum(float(x) for x in re.findall(r"^vllm:prefix_cache_hits_total ([\d.]+)$", b.decode(), re.M)) == 3.0


def test_key_and_errors(router):
    assert _req(router + "/v1/completions", {"model": "critic"}, key=None)[0] == 401
    assert _req(router + "/v1/models", key="wrong")[0] == 401
    code, b = _req(router + "/v1/completions", {"model": "boom"})
    assert code == 400 and json.loads(b)["error"] == "bad model"
    assert _req(router + "/health", key=None)[0] == 200
