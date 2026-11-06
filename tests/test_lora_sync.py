"""双机 GRPO 的 LoRA 同步：密钥校验、拷贝到内存盘目录、Server.load_lora / unload_lora 走同步服务（dry：不调 vLLM）。"""
import threading
from http.server import ThreadingHTTPServer

import json
from urllib import error, request

import pytest

from scripts import lora_sync
from train import gen_grpo


@pytest.fixture()
def sync_server(tmp_path):
    shm = tmp_path / "shm"
    shm.mkdir()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), lora_sync.make_handler("http://127.0.0.1:9/v1", "k3y", shm, dry=True))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", shm
    srv.shutdown()


def _post(url, body, key=None):
    h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    try:
        with request.urlopen(request.Request(url, data=json.dumps(body).encode(), headers=h, method="POST"), timeout=10) as r:
            return r.status
    except error.HTTPError as e:
        return e.code


def test_rejects_without_key(sync_server):
    url, _ = sync_server
    assert _post(url + "/sync", {"name": "p", "src": "/"}) == 401
    assert _post(url + "/sync", {"name": "p", "src": "/"}, key="wrong") == 401
    assert _post(url + "/sync", {"name": "../x", "src": "/"}, key="k3y") == 400


def test_server_load_and_unload_go_through_sync(sync_server, tmp_path, monkeypatch):
    pytest.importorskip("requests")
    pytest.importorskip("openai")
    url, shm = sync_server
    monkeypatch.setenv("VLLM_API_KEY", "k3y")
    monkeypatch.setenv("LORA_SYNC_URL", url)
    src = tmp_path / "fs" / "lora" / "v1" / "policy"
    src.mkdir(parents=True)
    (src / "adapter_model.safetensors").write_bytes(b"0" * 4096)
    (src / "adapter_config.json").write_text("{}")
    s = gen_grpo.Server("http://127.0.0.1:9/v1")      # vLLM 地址不会被直接访问
    s.load_lora("policy_v1", str(src))
    assert (shm / "policy_v1" / "adapter_model.safetensors").stat().st_size == 4096
    assert s.last_sync["name"] == "policy_v1"
    s.unload_lora("policy_v1")
    assert not (shm / "policy_v1").exists()


def test_key_defaults_to_x_without_env(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("LORA_SYNC_URL", raising=False)
    assert gen_grpo.vllm_key() == "x" and gen_grpo.lora_sync_url() == ""


def test_upload_push_extracts_and_rejects_bad_tar(sync_server, tmp_path):
    import io
    import tarfile
    url, shm = sync_server

    def put(name, data, key="k3y"):
        req = request.Request(f"{url}/push/{name}", data=data, method="PUT", headers={"Authorization": f"Bearer {key}"})
        try:
            with request.urlopen(req, timeout=10) as r:
                return r.status
        except error.HTTPError as e:
            return e.code

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:") as tf:
        for n, b in (("adapter_model.safetensors", b"1" * 2048), ("adapter_config.json", b"{}")):
            ti = tarfile.TarInfo(n)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    assert put("policy_v2", buf.getvalue(), key="bad") == 401
    assert put("policy_v2", buf.getvalue()) == 200
    assert (shm / "policy_v2" / "adapter_model.safetensors").stat().st_size == 2048
    evil = io.BytesIO()
    with tarfile.open(fileobj=evil, mode="w:") as tf:
        ti = tarfile.TarInfo("../escape.txt")
        ti.size = 1
        tf.addfile(ti, io.BytesIO(b"x"))
    assert put("policy_v3", evil.getvalue()) == 400
    assert not (shm.parent / "escape.txt").exists()


def test_server_upload_mode(sync_server, tmp_path, monkeypatch):
    pytest.importorskip("requests")
    pytest.importorskip("openai")
    url, shm = sync_server
    monkeypatch.setenv("VLLM_API_KEY", "k3y")
    monkeypatch.setenv("LORA_SYNC_URL", url)
    monkeypatch.setenv("LORA_SYNC_MODE", "upload")
    src = tmp_path / "local" / "v4"
    src.mkdir(parents=True)
    (src / "adapter_model.safetensors").write_bytes(b"2" * 1000)
    (src / "adapter_config.json").write_text("{}")
    s = gen_grpo.Server("http://127.0.0.1:9/v1")
    s.load_lora("policy_v4", str(src))
    assert (shm / "policy_v4" / "adapter_model.safetensors").read_bytes() == b"2" * 1000


class _FakeModel:
    def save_pretrained(self, d, selected_adapters=None):
        from pathlib import Path
        Path(d).mkdir(parents=True, exist_ok=True)
        (Path(d) / "adapter_model.safetensors").write_bytes(b"w")


class _FakeServer:
    def __init__(self, fail_at=None):
        self.loaded, self.unloaded, self.fail_at = [], [], fail_at

    def load_lora(self, name, path):
        if name == self.fail_at:
            raise RuntimeError("upload failed")
        self.loaded.append(name)

    def unload_lora(self, name):
        self.unloaded.append(name)


def test_async_pusher_updates_version_and_unloads_old(tmp_path):
    from types import SimpleNamespace
    from train.ans_grpo import AsyncPusher
    prod, srv = SimpleNamespace(version=0), _FakeServer()
    p = AsyncPusher([srv], prod, keep_versions=2)
    for v in (1, 2, 3):
        p.push(_FakeModel(), tmp_path, v)
    p.wait()
    assert prod.version == 3 and srv.loaded == ["policy_v1", "policy_v2", "policy_v3"] and srv.unloaded == ["policy_v1"]
    assert not (tmp_path / "lora" / "v1").exists() and (tmp_path / "lora" / "v3").exists()


def test_async_pusher_raises_and_keeps_old_version(tmp_path):
    from types import SimpleNamespace
    from train.ans_grpo import AsyncPusher
    prod = SimpleNamespace(version=0)
    p = AsyncPusher([_FakeServer(fail_at="policy_v2")], prod, keep_versions=2)
    p.push(_FakeModel(), tmp_path, 1)
    p.push(_FakeModel(), tmp_path, 2)
    with pytest.raises(RuntimeError, match="async LoRA push failed"):
        p.wait()
    assert prod.version == 1      # 失败的版本不会被采样端用上
