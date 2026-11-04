"""释放（永久删除）已关机的 AutoDL 实例。只接受显式给出的完整 uuid；主实例永不释放，未关机的实例跳过。

用法：python -m scripts.autodl_release <uuid> [<uuid> ...]
"""
from __future__ import annotations

import argparse

from scripts.autodl_web import Web

MAIN = "c5k0nax3ry"   # 主实例，按约定保留


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("uuids", nargs="+")
    ap.add_argument("--session", default="autodl_web.json")
    args = ap.parse_args()
    web = Web(args.session)
    status = {i["uuid"]: i.get("status") for i in web.instances()}
    for u in args.uuids:
        if u.startswith(MAIN):
            print(u, "skip: main instance")
        elif u not in status:
            print(u, "skip: not found")
        elif status[u] != "shutdown":
            print(u, f"skip: status {status[u]}, only shut-down instances are released")
        else:
            r = web.call("/api/v1/instance/release", {"instance_uuid": u})
            print(u, r.get("code"), r.get("msg"))


if __name__ == "__main__":
    main()
