"""给已有的 AutoDL 实例抢开机；原主机一直没卡时克隆到同地区另一台有空闲卡的主机。成功后刷新本地登录信息并通知。

关机的实例绑定在原主机上，主机没有空闲卡时开机返回「机器空闲GPU不足」。等 clone_after_min 分钟仍开不了机，
就在同一数据中心找有空闲 GPU、在线、价格不超过 max_price 的主机，用网页的「克隆实例」接口
（/api/v1/order/instance/clone/payg，字段参考 github.com/Fo-OLLL/autodl-card-helper）复制到新主机，
并复制数据盘（copy_data_disk_after_clone）。原实例保持关机不释放，确认新实例可用后再在控制台手动释放。

用法：python -m scripts.autodl_poweron --uuid 611b489072-8df08a9e --max_hours 12 --clone_after_min 20
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scripts.autodl_grab import log, notify
from scripts.autodl_web import GIB, Web, usable

KEEP = ("uuid", "name", "status", "region_name", "machine_id", "ssh_command", "proxy_host", "ssh_port",
        "root_password", "jupyter_domain")


def find(web: Web, uuid: str) -> dict | None:
    return next((i for i in web.instances() if i["uuid"] == uuid), None)


def clone_body(web: Web, src: str, machine_id: str, name: str) -> dict:
    snap = web.call(f"/api/v1/instance/snapshot?instance_uuid={src}").get("data") or {}
    img = snap.get("image") or {}
    expand = snap.get("expand_data_disk_size") or 0
    cg_info = snap.get("cg_application_info") or {"image_name": "", "current_version_id": 0, "current_version": "",
                                                    "image_id": 0}
    return {
        "instance_uuid": src,
        "instance_info": {
            "machine_id": machine_id, "charge_type": "payg", "req_gpu_amount": 1,
            "base_image_uuid": img.get("base_image_uuid") or "", "private_image_uuid": img.get("private_image_uuid") or "",
            "reproduction_uuid": img.get("reproduction_uuid") or "", "reproduction_id": img.get("reproduction_id") or 0,
            "cg_application_uuid": snap.get("cg_application_uuid") or "", "cg_application_info": cg_info,
            "instance_name": name, "expand_data_disk": expand,
            "copy_data_disk_after_clone": True, "keep_src_user_service_address_after_clone": False, "ft_sparse": False},
        "price_info": {"coupon_id_list": [], "machine_id": machine_id, "charge_type": "payg", "duration": 1, "num": 1,
                       "expand_data_disk": expand},
    }


def try_clone(web: Web, src: dict, gpu: str, max_price: float, logf: Path) -> str | None:
    """返回新实例 uuid（下单接口 data 字段；克隆出的实例名可能为空，不能按名字找）；没有合适主机返回 None。"""
    snap = web.call(f"/api/v1/instance/snapshot?instance_uuid={src['uuid']}").get("data") or {}
    expand = snap.get("expand_data_disk_size") or 0
    dc = next((r["data_center"] for r in web.call("/api/v1/region/list").get("data") or []
               if r.get("region_name") == src.get("region_name")), "westDC3")
    signs = web.region_signs(dc)
    cands = [m for m in web.machines(signs, gpu) if usable(m, expand, max_price) and m["machine_id"] != src["machine_id"]]
    for m in cands:
        name = f"CLONE-{time.strftime('%m%d%H%M%S')}"
        r = web.call("/api/v1/order/instance/clone/payg", clone_body(web, src["uuid"], m["machine_id"], name))
        if r.get("code") == "Success":
            log(f"clone ordered on {m['machine_id']} ({m.get('payg_price', 0) / 1000:.2f}/h): {r.get('data')}", logf)
            return r.get("data") if isinstance(r.get("data"), str) else name
        log(f"clone on {m['machine_id']} failed: {r.get('code')} {r.get('msg')}", logf)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uuid", required=True)
    ap.add_argument("--session", default="autodl_web.json")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--max_hours", type=float, default=12)
    ap.add_argument("--clone_after_min", type=float, default=20, help="原主机开不了机多久后克隆到别的主机；<0 不克隆")
    ap.add_argument("--gpu", default="RTX 5090")
    ap.add_argument("--max_price", type=float, default=5.0)
    ap.add_argument("--notify", nargs="*", default=["email", "toast"])
    ap.add_argument("--out_dir", default="outputs/autodl")
    args = ap.parse_args()

    out = Path(args.out_dir)
    logf = out / "poweron.log"
    web = Web(args.session)
    t0, deadline, tries = time.time(), time.time() + args.max_hours * 3600, 0
    target, clone_name = args.uuid, None
    while time.time() < deadline:
        tries += 1
        if clone_name:   # 等克隆出的新实例开机、数据盘复制完成
            inst = next((i for i in web.instances() if clone_name in (i["uuid"], i.get("name"))), None)
            if inst and inst["status"] == "running":
                copy = (inst.get("additional_info") or {}).get("copy_data_disk_after_clone_status")
                if copy in (None, "", "Succeed", "Success"):
                    target = inst["uuid"]
                    break
                log(f"clone running, data disk copy: {copy}", logf)
            elif inst and inst["status"] in ("create_failed", "created_clone_failed"):
                log(f"clone failed: {inst['status']}", logf)
                clone_name = None
        else:
            src = find(web, args.uuid)
            if src["status"] == "running":
                target = src["uuid"]
                break
            if src["status"] == "shutdown":
                r = web.call("/api/v1/instance/power_on", {"instance_uuid": args.uuid})
                if r.get("code") == "Success":
                    log(f"power_on accepted after {tries} tries", logf)
                elif tries == 1 or tries % 30 == 0:
                    log(f"try {tries}: {r.get('code')} {r.get('msg')}", logf)
                if (r.get("code") != "Success" and args.clone_after_min >= 0
                        and time.time() - t0 > args.clone_after_min * 60):
                    clone_name = try_clone(web, src, args.gpu, args.max_price, logf)
        time.sleep(args.interval)
    else:
        notify("AutoDL 开机失败", f"{args.max_hours} 小时内既没开机也没克隆成功", args.notify, logf)
        raise SystemExit(1)

    inst = find(web, target)
    (out / "web_instance.json").write_text(json.dumps({k: inst.get(k) for k in KEEP}, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    how = f"克隆到新主机 {inst.get('machine_id')}（原实例 {args.uuid} 保持关机）" if target != args.uuid else "原实例开机"
    notify("AutoDL 5090 已就绪", f"{how}，uuid {target}，开始跑实验", args.notify, logf)
    print(f"READY {target}")


if __name__ == "__main__":
    main()
