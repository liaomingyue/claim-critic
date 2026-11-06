"""采样 / 打分放子进程（ProcProducer）：跨进程的批次队列、共享版本号、prompt 位置、子进程出错、收尾。用假的子进程，不需要 vLLM。"""
import time
from types import SimpleNamespace

import pytest


def fake_child(args, out, start_pos, vval, bq, cmd, ack):
    """每批带当前共享版本号和位置；收到 stop 回 done。args.die_after > 0 时出那么多批后报错。"""
    import queue
    pos, n = start_pos, 0
    while True:
        try:
            c = cmd.get_nowait()
            if c[0] == "stop":
                ack.put("done")
                return
        except queue.Empty:
            pass
        if args.die_after and n >= args.die_after:
            bq.put({"__dead__": "RuntimeError('boom')"})
            time.sleep(0.2)
            continue
        pos += 16
        try:
            bq.put({"version": vval.value, "samples": [{"x": 1}] * 4, "pos": pos}, timeout=0.2)
            n += 1
        except queue.Full:
            pos -= 16


def _prod(tmp_path, die_after=0):
    from train.ans_grpo import ProcProducer
    p = ProcProducer(SimpleNamespace(max_stale=2, version_prefix="m_", die_after=die_after), tmp_path)
    p.target = fake_child
    return p


def test_batches_version_and_pos(tmp_path):
    p = _prod(tmp_path)
    p.pos = 100          # 续训：load_state 设好位置再启动
    p.start()
    b1 = p.get()
    assert b1["version"] == 0 and p.pos == b1["pos"] and b1["pos"] > 100
    p.version = 7        # 主进程（后台推送）更新版本，子进程之后的批次跟着变
    deadline = time.time() + 10
    while time.time() < deadline:
        if p.get()["version"] == 7:
            break
    else:
        pytest.fail("version update not seen by child")
    p.shutdown(timeout=10)
    assert not p.p.is_alive()


def test_child_error_raises_in_main(tmp_path):
    p = _prod(tmp_path, die_after=1)
    p.start()
    p.get()
    with pytest.raises(RuntimeError, match="producer died"):
        for _ in range(5):
            p.get()
    p.p.terminate()
