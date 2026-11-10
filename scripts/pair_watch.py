"""双机看门狗（机器完成后关掉；GPU 利用率 10 分钟低于 10% 也关机）。
每台机器各自判断：每 interval 秒采一次所有卡的利用率，连续 idle_minutes 分钟都 < idle_util 就关这台。
训练跑完两台都会闲下来，自然在 idle_minutes 内关掉。本地跑：

  python -m scripts.pair_watch outputs/autodl/extra_pairtrain.json outputs/autodl/extra_pairinfer.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from scripts.autodl_extra import sh  # noqa: E402
from scripts.autodl_grab import log, notify  # noqa: E402
from scripts.autodl_web import Web  # noqa: E402


KEEPALIVE = "/root/autodl-tmp/KEEPALIVE"   # 脚本在不用 GPU 的长步骤（拷 9B、加载模型）里定期 touch（踩过：拷 9B 时 GPU 空闲被关机）


def gpu_util(info: dict) -> list[int] | None:
    """所有卡的利用率；KEEPALIVE 5 分钟内更新过就当忙（返回 [100]）。"""
    try:
        out = sh(info, f"nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits < /dev/null; "
                       f"[ -n \"$(find {KEEPALIVE} -mmin -5 2>/dev/null)\" ] && echo KEEPALIVE")
    except Exception:  # noqa: BLE001 网络抖动，下次再看
        return None
    if "KEEPALIVE" in out:
        return [100]
    u = [int(x) for x in out.split() if x.strip().isdigit()]
    return u or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("infos", nargs="+", help="outputs/autodl/extra_*.json")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--idle_minutes", type=float, default=10)
    ap.add_argument("--idle_util", type=int, default=10, help="所有卡利用率都低于这个算空闲")
    ap.add_argument("--notify", nargs="*", default=["email", "toast"])
    a = ap.parse_args()
    logf = Path("outputs/autodl/pair_watch.log")
    infos = {p: json.loads(Path(p).read_text(encoding="utf-8")) for p in a.infos}
    idle_since: dict[str, float | None] = {p: None for p in infos}
    alive = set(infos)
    log(f"pair_watch start: {[infos[p]['uuid'] for p in infos]} idle<{a.idle_util}% for {a.idle_minutes} min", logf)
    while alive:
        for p in sorted(alive):
            u = gpu_util(infos[p])
            if u is None:   # 连不上不算空闲也不算忙，保持原计时
                continue
            busy = any(x >= a.idle_util for x in u)
            idle_since[p] = None if busy else (idle_since[p] or time.time())
            if idle_since[p] and time.time() - idle_since[p] >= a.idle_minutes * 60:
                r = Web("autodl_web.json").call("/api/v1/instance/power_off", {"instance_uuid": infos[p]["uuid"]})
                log(f"power_off {infos[p]['uuid']} ({p}): {r.get('code')} {r.get('msg')}", logf)
                notify("AutoDL 已关机", f"{infos[p]['uuid']}：GPU 利用率连续 {a.idle_minutes:g} 分钟 < {a.idle_util}%；关机 {r.get('code')}", a.notify, logf)
                alive.discard(p)
        time.sleep(a.interval)
    log("pair_watch done", logf)


if __name__ == "__main__":
    main()
