"""多开几台 5090 分担主实例排队的实验：抢到卡就建新实例，从主实例经同地区网络 rsync 拉环境、模型和数据，再跑分给它的那部分。

角色按顺序分配（见 scripts/remote/bootstrap_worker.sh）：
  diag ：格式诊断（e4、9B 只判定）+ 9B 合成实验 direct 组；
  syn9b：9B 判识破率 + 9B 合成实验 critic / random 组。
某个角色交给新机器后，主实例上还在排队（还没开始）的同一部分会被停掉；已经开始的不动（新机器也就不接这个角色）。

新实例用和主实例相同的基础镜像（conda 环境路径一致，整份 rsync 过来不用重装）。
登录信息写 outputs/autodl/extra_<角色>.json，用 AUTODL_INFO=该文件 python -m scripts.remote ... 操作那台机器。

--fs 模式（2026-10-28 起）：新实例用自定义镜像（--image，环境已在系统盘）+ 文件存储（模型、代码和数据共享），
不经过主实例：上传本地最新代码包后运行 scripts/remote/bootstrap_fs.sh <角色> <运行名>，结果写回文件存储。

用法：python -m scripts.autodl_extra --roles diag syn9b
      python -m scripts.autodl_extra --fs --image <镜像 uuid> --roles closedloop
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import tarfile
import time
from pathlib import Path

import paramiko

from scripts.autodl_grab import log, notify
from scripts.autodl_poweron import KEEP
from scripts.autodl_web import GIB, Web, usable

MAIN = "outputs/autodl/web_instance.json"
D = "/root/autodl-tmp/critic_model"
QUEUED = {"diag": ("run_diag_exp", "diag.log"), "syn9b": ("run_synth9b_exp", "syn9b.log")}  # 主实例上对应的排队脚本和日志
UPLOAD = ["scripts/remote/bootstrap_worker.sh", "scripts/remote/run_diag_exp.sh", "scripts/remote/run_synth9b_exp.sh",
          "scripts/remote/run_iter.sh", "scripts/remote/run_pools.sh", "scripts/remote/run_discrim_head.sh", "scripts/remote/run_gan_critic.sh",
          "scripts/remote/run_realgen.sh",
          "scripts/remote/run_closedloop.sh"]


def ssh(info: dict) -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(info["proxy_host"], port=info["ssh_port"], username="root", password=info["root_password"], timeout=30)
    return c


def sh(info: dict, cmd: str, timeout: int = 120) -> str:
    c = ssh(info)
    _, o, e = c.exec_command(f"bash -lc {json.dumps(cmd)}", timeout=timeout)
    out = o.read().decode(errors="replace") + e.read().decode(errors="replace")
    c.close()
    return out


def put(info: dict, local: str, remote: str) -> None:
    c = ssh(info)
    tmp = "/root/" + remote.rsplit("/", 1)[-1] + ".part"
    s = c.open_sftp()
    s.put(local, tmp)
    s.close()
    c.exec_command(f"mkdir -p {os.path.dirname(remote)} && mv {tmp} {remote}")[1].channel.recv_exit_status()
    c.close()


def claim_role(main: dict, role: str, logf: Path) -> bool:
    """主实例上这部分还没开始就停掉它，返回 True（交给新机器）；已经开始了返回 False。"""
    if role not in QUEUED:   # 主实例上没有排这部分（如迭代实验的随机序列），直接交给新机器
        return True
    proc, lg = QUEUED[role]
    out = sh(main, f"if [ -s /root/autodl-tmp/{lg} ]; then echo STARTED; else pkill -f '[{proc[0]}]{proc[1:]}'; echo CLAIMED; fi")
    log(f"claim {role} on main: {out.strip()}", logf)
    # 停进程和重启分两次调用：重启命令里带着脚本路径，和 pkill 放在同一条命令里会被 pkill -f 匹配到、连自己一起杀掉
    if "CLAIMED" in out and role == "diag":
        # diag 角色顺带跑 9B direct 组：主实例上还没开始的 9B 合成实验改成只跑 critic / random
        p, l9 = QUEUED["syn9b"]
        r = sh(main, f"if [ ! -s /root/autodl-tmp/{l9} ] && pgrep -f '[{p[0]}]{p[1:]}' >/dev/null; then "
                     f"pkill -f '[{p[0]}]{p[1:]}'; echo KILLED; fi")
        if "KILLED" in r:
            bg(main, f"ARMS='critic random' bash {D}/scripts/remote/run_synth9b_exp.sh", "/root/autodl-tmp/syn9b_runner.log")
        log(f"main syn9b without direct: {r.strip() or 'not queued'}", logf)
    if "CLAIMED" in out and role == "syn9b":
        # 新机器只跑 critic / random；诊断还留在主实例上时，direct 组跟在诊断后面由主实例跑
        if "DIAG" in sh(main, "pgrep -f '[r]un_diag_exp' >/dev/null && echo DIAG"):
            bg(main, f"ARMS=direct bash {D}/scripts/remote/run_synth9b_exp.sh", "/root/autodl-tmp/syn9b_runner.log")
            log("main syn9b: direct arm queued after diag", logf)
    return "CLAIMED" in out


def bg(info: dict, cmd: str, logfile: str) -> None:
    """后台启动（同 scripts.remote --bg）：setsid、不分配 pty、不套 bash -lc，SSH 断开后进程不被一起杀掉。"""
    c = ssh(info)
    c.exec_command(f"setsid nohup bash -lc {json.dumps(cmd)} >> {logfile} 2>&1 < /dev/null &")[1].read()
    c.close()


FS_IMAGE = "image-8abf4719aa"   # 主实例系统盘（含 conda 环境）保存的自定义镜像，2026-10-28；镜像不可用时 bootstrap_fs 从文件存储解压环境包
CODE_DIRS = ["common", "ragtruth", "train", "eval", "infer", "scripts", "configs"]   # 不含 outputs 和本地密钥文件


def code_tarball() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for d in CODE_DIRS + ["pyproject.toml"]:
            if Path(d).exists():
                tar.add(d, filter=lambda t: None if "__pycache__" in t.name or t.name.endswith(".pyc") else t)
    return buf.getvalue()


def setup_fs(new: dict, role: str, run: str, logf: Path) -> None:
    """--fs 模式：传最新代码包和 bootstrap_fs.sh，后台启动；不碰主实例。"""
    tmp = Path("outputs/autodl/code_update.tar")
    tmp.write_bytes(code_tarball())
    put(new, str(tmp), "/root/autodl-tmp/code_update.tar")
    put(new, "scripts/remote/bootstrap_fs.sh", "/root/autodl-tmp/bootstrap_fs.sh")
    sh(new, "sed -i 's/\\r$//' /root/autodl-tmp/bootstrap_fs.sh")
    bg(new, f"bash /root/autodl-tmp/bootstrap_fs.sh {role} {run}", "/root/autodl-tmp/bootstrap.log")
    log(f"{role}: bootstrap_fs started on {new['uuid']} (run {run}, code {tmp.stat().st_size // 1024} KB)", logf)


def setup(main: dict, new: dict, role: str, logf: Path) -> None:
    key = sh(new, "test -f ~/.ssh/id_ed25519 || ssh-keygen -q -t ed25519 -N '' -f ~/.ssh/id_ed25519; cat ~/.ssh/id_ed25519.pub").strip()
    sh(main, f"mkdir -p ~/.ssh && grep -qF {json.dumps(key)} ~/.ssh/authorized_keys 2>/dev/null || echo {json.dumps(key)} >> ~/.ssh/authorized_keys")
    sh(new, f"mkdir -p {D}/scripts/remote")
    for f in UPLOAD:
        put(new, f, f"{D}/{f}")
    sh(new, f"sed -i 's/\\r$//' {D}/scripts/remote/*.sh")
    # 拉完代码目录会覆盖上面上传的脚本：先把最新脚本传到主实例，两边一致
    for f in UPLOAD:
        put(main, f, f"{D}/{f}")
    sh(main, f"sed -i 's/\\r$//' {D}/scripts/remote/*.sh")
    bg(new, f"bash {D}/scripts/remote/bootstrap_worker.sh {main['proxy_host']} {main['ssh_port']} {role}",
       "/root/autodl-tmp/bootstrap.log")
    log(f"{role}: bootstrap started on {new['uuid']}", logf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", nargs="+", default=["diag", "syn9b"])
    ap.add_argument("--session", default="autodl_web.json")
    ap.add_argument("--region", default="westDC3")
    ap.add_argument("--gpu", nargs="+", default=["RTX 5090", "vGPU-48GB"],
                    help="可接受的卡型，按顺序优先（vGPU-48GB 显存 48G、单价和 5090 相当；4090 只有 24G，放不下 35B）")
    ap.add_argument("--expand_gb", type=int, default=100, help="环境 12G + 模型和权重约 40G + 新训练的权重，默认 50G 免费盘 + 100G")
    ap.add_argument("--max_price", type=float, default=5.0)
    ap.add_argument("--gpus", type=int, default=1, help="每台机器要几张卡（GRPO：训练卡 + 推理卡 = 2）")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--max_hours", type=float, default=6, help="超过这么久还没抢到就不再开（主实例排队也快跑完了）")
    ap.add_argument("--notify", nargs="*", default=["email", "toast"])
    ap.add_argument("--exclude_machines", nargs="*", default=[], help="不要开在这些主机上（如网速慢的主机）")
    ap.add_argument("--fs", action="store_true", help="自定义镜像 + 文件存储模式（bootstrap_fs.sh），不经过主实例")
    ap.add_argument("--image", default="", help="用这个镜像 uuid 创建；空：--fs 时用自定义镜像 FS_IMAGE，否则主实例的基础镜像")
    ap.add_argument("--expand_gb_fs", type=int, default=60, help="--fs 模式的数据盘扩容：只放模型副本和新训练的权重")
    args = ap.parse_args()

    logf = Path("outputs/autodl/extra.log")
    web = Web(args.session)
    main_info = json.loads(Path(MAIN).read_text(encoding="utf-8"))
    image = args.image or (FS_IMAGE if args.fs else "")
    if not image:
        snap = web.call(f"/api/v1/instance/snapshot?instance_uuid={main_info['uuid']}").get("data") or {}
        image = (snap.get("image") or {}).get("base_image_uuid")
    if not image:
        raise SystemExit("主实例的基础镜像查不到")
    expand = (args.expand_gb_fs if args.fs else args.expand_gb) * GIB
    signs = web.region_signs(args.region)
    todo = list(args.roles)
    deadline = time.time() + args.max_hours * 3600
    log(f"want {todo}, image {image}", logf)
    while todo and time.time() < deadline:
        try:   # 偶发网络超时不能让整个抢机器循环退出（2026-11-04 踩过：抢了 2.5 小时后一次 timed out 就退了）
            cands = [m for g in args.gpu for m in web.machines(signs, g)
                     if usable(m, expand, args.max_price, args.gpus) and m["machine_id"] not in args.exclude_machines]
        except Exception as e:  # noqa: BLE001
            log(f"machine list error, retry: {str(e)[:120]}", logf)
            time.sleep(args.interval)
            continue
        if not cands:
            time.sleep(args.interval)
            continue
        role = todo[0]
        name = f"WORKER-{role}-{time.strftime('%m%d%H%M')}"
        before = {i["uuid"] for i in web.instances()}
        r = web.create(cands[0]["machine_id"], image, expand, name, args.gpus)
        log(f"create {name} on {cands[0]['machine_id']}: {r.get('code')} {r.get('msg')}", logf)
        if r.get("code") != "Success":
            if image.startswith("image-") and "镜像" in str(r.get("msg")):   # 自定义镜像用不了：退回基础镜像（环境从文件存储解压）
                snap = web.call(f"/api/v1/instance/snapshot?instance_uuid={main_info['uuid']}").get("data") or {}
                image = (snap.get("image") or {}).get("base_image_uuid") or image
                log(f"custom image rejected; falling back to base image {image}", logf)
                continue
            time.sleep(args.interval)
            continue
        for _ in range(60):   # 等实例 running
            # 新建实例的 name 字段可能为空，按「下单前没有的实例」找
            inst = next((i for i in web.instances() if i["uuid"] not in before and i.get("status") not in ("released",)), None)
            if inst and inst["status"] == "running":
                break
            time.sleep(15)
        else:
            log(f"{name} did not reach running", logf)
            continue
        new = {k: inst.get(k) for k in KEEP}
        Path(f"outputs/autodl/extra_{role}.json").write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(30)   # sshd 刚起来
        if args.fs:
            if role == "pairinfer":   # 双机：推理机的对外域名只有平台接口查得到，建好就写进共享盘（不依赖本地会话，2026-11-04 踩过：会话断了训练机空等 1 小时）
                dom = {"svc6006": inst.get("service_6006_domain"), "svc6008": inst.get("service_6008_domain"), "infer_uuid": inst["uuid"]}
                # 先等共享盘挂上再写（2026-11-04 踩过：容器刚起时还没挂，写进了被挂载盖住的本地目录，训练机读到旧推理机的域名，404），写完读回核对
                got = sh(new, "for i in $(seq 60); do [ -d /root/autodl-fs/models ] && break; sleep 5; done; "
                              "P=/root/autodl-fs/runs/pair/$(cat /root/autodl-fs/runs/pair/CURRENT 2>/dev/null || echo default); mkdir -p $P && rm -f $P/infer_ready $P/train_done $P/infer_failed"
                              f" && echo {shlex.quote(json.dumps(dom))} > $P/domains.json && cat $P/domains.json")
                if inst["uuid"] not in got:
                    raise SystemExit(f"pairinfer: domains.json 写入核对失败：{got[:200]}")
                log(f"pairinfer: domains written and verified {dom}", logf)
            setup_fs(new, role, f"{role}-{time.strftime('%m%d%H%M')}", logf)
            notify(f"AutoDL 加开机器：{role}", f"新实例 {new['uuid']}（{cands[0]['machine_id']}）用镜像 + 文件存储起步，"
                                             f"跑 {role}", args.notify, logf)
            todo.pop(0)
            continue
        if not claim_role(main_info, role, logf):
            log(f"{role} already started on main; worker {new['uuid']} left idle — shutting it down", logf)
            web.call("/api/v1/instance/power_off", {"instance_uuid": new["uuid"]})
            todo.pop(0)
            continue
        setup(main_info, new, role, logf)
        # 这台机器自己的监控单独起（不做成本进程的子进程：本进程重启时子进程会被一起杀掉）：
        # AUTODL_INFO=outputs/autodl/extra_<role>.json python -m scripts.autodl_watch --uuid <uuid>         #   --exp_log /root/autodl-tmp/worker.log --done WORKER_DONE --proc bootstrap_worker --keep_on_file none
        notify(f"AutoDL 加开机器：{role}", f"新实例 {new['uuid']}（{cands[0]['machine_id']}）开始从主实例拉数据，"
                                         f"之后跑 {role} 部分", args.notify, logf)
        todo.pop(0)
    log(f"done, remaining roles {todo}", logf)


if __name__ == "__main__":
    main()
