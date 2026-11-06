"""双机 GRPO 配对（2026-11-04）：本地跑。推理机实例的对外服务域名只有 AutoDL 接口查得到（容器里拿不到），
这里查到后写到共享盘 runs/pair/<pair>/domains.json，并把 runs/pair/CURRENT 指向这个配对；训练机（run_train_node.sh）等这个文件。
登录信息读 outputs/autodl/<info>.json，不打印。

  python -m scripts.pair_setup --pair p1104 --via <任一台已开机实例的 json>                         # 抢机器前：只写 CURRENT
  python -m scripts.pair_setup --pair p1104 --infer_uuid <推理机 uuid> --via outputs/autodl/extra_pairinfer.json   # 推理机建好后：写域名
"""
from __future__ import annotations

import argparse
import json
import shlex

import paramiko

from scripts.autodl_web import Web


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--infer_uuid", default="", help="空：只把 CURRENT 指向这个配对（抢机器前先写，推理机开机时要读）")
    ap.add_argument("--via", required=True, help="用哪台实例的登录信息写共享盘（两台都挂着同一个共享盘）")
    ap.add_argument("--session", default="autodl_web.json")
    a = ap.parse_args()
    dom = None
    if a.infer_uuid:
        inst = next(i for i in Web(a.session).instances() if i["uuid"] == a.infer_uuid)
        dom = {"svc6006": inst["service_6006_domain"], "svc6008": inst["service_6008_domain"], "infer_uuid": a.infer_uuid}
        assert dom["svc6006"] and dom["svc6008"], "实例没有自定义服务域名"
    d = json.load(open(a.via, encoding="utf-8"))
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(d["proxy_host"], port=int(d["ssh_port"]), username="root", password=d["root_password"], timeout=30)
    P = f"/root/autodl-fs/runs/pair/{a.pair}"
    cmd = f"mkdir -p {P} && echo {shlex.quote(a.pair)} > /root/autodl-fs/runs/pair/CURRENT"
    if dom:
        cmd += f" && echo {shlex.quote(json.dumps(dom))} > {P}/domains.json && cat {P}/domains.json"
    _, o, e = c.exec_command(cmd, timeout=60)
    print("PAIR_SETUP", (o.read() + e.read()).decode().strip())


if __name__ == "__main__":
    main()
