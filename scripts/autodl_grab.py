"""AutoDL 抢卡：轮询创建「容器实例 Pro」（1 张 RTX 5090），抢到后检查价格、记录登录信息并通知。

全部走官方开放 API（https://www.autodl.com/docs/instance_pro_api/），开发者 token 从 autodl_token.txt 读取
（控制台 → 账号 → 设置 → 开发者Token），不打印、不写日志。

流程：
1. 先查实例列表：已有未释放的实例就不再创建（同一时间最多 1 台，防止重复扣费）。
2. 按 interval 秒轮询创建：1 × 5090（gpu_spec_uuid=5090-p）、系统盘扩容 200 GB、CUDA ≥ 12.8、
   地区 westDC3（西北B区）。没卡时创建会失败，记日志后继续；成功就停止轮询。
   不用弹性部署的库存接口判断：它查的是弹性部署的库存，不代表 Pro 实例能否创建。
3. 等实例进入 running，读详情：每小时价格超过 max_price 就关机并释放；否则把 SSH 信息写到
   outputs/autodl/instance.json（不进推送），通知只发「已抢到、实例 ID、价格」。
4. 到 max_hours 仍没抢到就退出。

镜像：Pro API 的公共 PyTorch 镜像最高只有 torch 2.0 + CUDA 11.8，跑不了 5090（Blackwell 需要 CUDA 12.8+）。
第一次用 Miniconda 基础镜像开机，再用 scripts/autodl_setup.sh 装环境并存成私有镜像；之后 --image 传私有镜像 uuid。

用法：
  python -m scripts.autodl_grab --dry_run                    # 只查列表和库存，不创建
  python -m scripts.autodl_grab --test_notify                # 发一条测试邮件 + 弹窗
  python -m scripts.autodl_grab --max_price 3.0
  python -m scripts.autodl_grab --image image-xxxx --max_price 3.0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.autodl.com"
GPU_SPEC = "5090-p"
MINICONDA_IMAGE = "base-image-mbr2n4urrc"     # Miniconda, CUDA 11.6, Ubuntu 20.04；torch 由 autodl_setup.sh 装 cu128 版
REGIONS = ["westDC3"]                     # 西北B区
ACTIVE = {"running", "starting", "creating", "booting", "re_initializing", "shutting_down"}


def log(msg: str, path: Path) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class AutoDL:
    def __init__(self, token: str):
        self.token = token

    def call(self, method: str, path: str, body: dict | None = None, params: str = "") -> dict:
        req = urllib.request.Request(API + path + params, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": self.token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return {"code": f"HTTP{e.code}", "msg": e.read()[:300].decode("utf-8", "replace")}
        except (urllib.error.URLError, TimeoutError) as e:
            return {"code": "NetworkError", "msg": str(e)}

    def instances(self) -> list[dict]:
        r = self.call("POST", "/api/v1/dev/instance/pro/list", {"page_index": 1, "page_size": 100})
        return (r.get("data") or {}).get("list") or []

    def stock(self, region: str) -> dict:
        r = self.call("POST", "/api/v1/dev/machine/region/gpu_stock", {"region_sign": region})
        return {k: v for d in r.get("data") or [] for k, v in d.items()}

    def create(self, image: str, disk_gb: int, name: str) -> dict:
        return self.call("POST", "/api/v1/dev/instance/pro/create", {
            "data_center_list": REGIONS, "req_gpu_amount": 1, "expand_system_disk_by_gb": disk_gb,
            "gpu_spec_uuid": GPU_SPEC, "image_uuid": image, "cuda_v_from": 128, "instance_name": name})

    def status(self, uuid: str) -> str:
        return str(self.call("GET", "/api/v1/dev/instance/pro/status", params=f"?instance_uuid={uuid}").get("data"))

    def snapshot(self, uuid: str) -> dict:
        return self.call("GET", "/api/v1/dev/instance/pro/snapshot", params=f"?instance_uuid={uuid}").get("data") or {}

    def power_off(self, uuid: str) -> dict:
        return self.call("POST", "/api/v1/dev/instance/pro/power_off", {"instance_uuid": uuid})

    def release(self, uuid: str) -> dict:
        return self.call("POST", "/api/v1/dev/instance/pro/release", {"instance_uuid": uuid})


def send_email(title: str, body: str, cfg_path: str = "smtp.json") -> None:
    """邮件通知。smtp.json（本地，不进仓库）：
    {"host": "smtp.qq.com", "port": 465, "user": "xxx@qq.com", "password": "<邮箱授权码>", "to": "xxx@qq.com"}
    QQ 邮箱：设置 → 账号 → 开启 SMTP 服务 → 生成授权码；password 填授权码，不是登录密码。"""
    import smtplib
    from email.header import Header
    from email.mime.text import MIMEText
    from email.utils import formataddr, formatdate, make_msgid

    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header(title, "utf-8")
    # 补全发件人名称、Date、Message-ID：缺这些头时 Gmail 更容易判成垃圾邮件
    msg["From"] = formataddr((str(Header("AutoDL 抢卡", "utf-8")), cfg["user"]))
    # 同时发给 to 和发件邮箱自己：QQ → Gmail 时常被拦截或延迟，自己发给自己不会丢
    rcpts = list(dict.fromkeys([cfg.get("to", cfg["user"]), cfg["user"]]))
    msg["To"] = ", ".join(rcpts)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=cfg["user"].split("@")[-1])
    with smtplib.SMTP_SSL(cfg["host"], int(cfg.get("port", 465)), timeout=30) as s:
        s.login(cfg["user"], cfg["password"])
        s.sendmail(cfg["user"], rcpts, msg.as_string())


def notify(title: str, body: str, methods: list[str], logf: Path) -> None:
    """email：发到自己的邮箱（见 send_email）；toast：Windows 右下角弹窗。通知内容不含登录密码。"""
    if "email" in methods:
        try:
            send_email(title, body)
        except Exception as e:  # noqa: BLE001 通知失败不影响实例本身，记日志
            log(f"[notify] email failed: {type(e).__name__}: {e}", logf)
    if "toast" in methods:  # Windows 原生通知（旧式托盘气泡在 Win11 上不弹出）
        ps = ("$null = [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
              "ContentType = WindowsRuntime]; "
              "$x = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
              "[Windows.UI.Notifications.ToastTemplateType]::ToastText02); "
              "$t = $x.GetElementsByTagName('text'); "
              f"$null = $t.Item(0).AppendChild($x.CreateTextNode('{title}')); "
              f"$null = $t.Item(1).AppendChild($x.CreateTextNode('{body}')); "
              "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
              "'{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe')"
              ".Show([Windows.UI.Notifications.ToastNotification]::new($x))")
        subprocess.Popen(["powershell", "-NoProfile", "-Command", ps])
    log(f"[notify] {title}: {body}", logf)


def wait_running(api: AutoDL, uuid: str, logf: Path, timeout: int = 900) -> str:
    t0, st = time.time(), ""
    while time.time() - t0 < timeout:
        st = api.status(uuid)
        if st == "running":
            return st
        time.sleep(10)
    log(f"instance {uuid} not running after {timeout}s (status {st})", logf)
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token_file", default="autodl_token.txt")
    ap.add_argument("--image", default=MINICONDA_IMAGE, help="镜像 uuid；装好环境后换成私有镜像")
    ap.add_argument("--disk_gb", type=int, default=200, help="系统盘扩容（Pro API 没有数据盘参数）")
    ap.add_argument("--max_price", type=float, required=False, help="每小时最高价（元），超过就释放")
    ap.add_argument("--interval", type=int, default=30, help="轮询间隔（秒）")
    ap.add_argument("--max_hours", type=float, default=24)
    ap.add_argument("--notify", nargs="*", default=["email", "toast"])
    ap.add_argument("--test_notify", action="store_true", help="只发一条测试通知")
    ap.add_argument("--name", default="critic-5090")
    ap.add_argument("--out_dir", default="outputs/autodl")
    ap.add_argument("--dry_run", action="store_true", help="只查实例列表和库存，不创建")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logf = out / "grab.log"
    if args.test_notify:
        notify("AutoDL 抢卡通知测试", "这是一条测试邮件：抢到 5090 时会收到同样格式的通知。", args.notify, logf)
        return
    api = AutoDL(Path(args.token_file).read_text(encoding="utf-8").strip())

    existing = [i for i in api.instances() if i.get("status") in ACTIVE]
    for i in api.instances():
        log(f"existing: {i.get('uuid')} {i.get('region_name')} {i.get('status')}", logf)
    for region in REGIONS:
        s = api.stock(region).get("RTX 5090", {})
        log(f"stock {region} RTX 5090 (弹性部署口径): idle {s.get('idle_gpu_num')} / {s.get('total_gpu_num')}", logf)
    if args.dry_run:
        return
    if args.max_price is None:
        raise SystemExit("--max_price 必填：抢到后价格超过它会自动释放")
    if existing:
        raise SystemExit(f"已有未释放的实例 {[i.get('uuid') for i in existing]}，不再创建")

    deadline, tries = time.time() + args.max_hours * 3600, 0
    while time.time() < deadline:
        tries += 1
        r = api.create(args.image, args.disk_gb, args.name)
        if r.get("code") == "Success" and r.get("data"):
            uuid = r["data"]
            log(f"created {uuid} after {tries} tries", logf)
            break
        if tries == 1 or tries % 20 == 0:
            log(f"try {tries}: {r.get('code')} {r.get('msg')} request_id={r.get('request_id')}", logf)
        time.sleep(args.interval)
    else:
        notify("AutoDL 未抢到 5090", f"{args.max_hours} 小时内 {tries} 次尝试都没有空闲", args.notify, logf)
        raise SystemExit(1)

    st = wait_running(api, uuid, logf)
    snap = api.snapshot(uuid)
    price = (snap.get("payg_price") or 0) / 1000      # API 金额单位为 元 × 1000
    if price > args.max_price:
        log(f"price {price:.2f}/h > max {args.max_price}; powering off and releasing {uuid}", logf)
        log(f"power_off: {api.power_off(uuid).get('code')}", logf)
        time.sleep(20)
        log(f"release: {api.release(uuid).get('code')}", logf)
        notify("AutoDL 5090 价格超限，已释放", f"{uuid} {price:.2f} 元/小时", args.notify, logf)
        raise SystemExit(2)

    info = {k: snap.get(k) for k in ("region_sign", "snapshot_gpu_alias_name", "ssh_command", "proxy_host",
                                     "ssh_port", "root_password", "jupyter_domain")}
    info.update({"uuid": uuid, "status": st, "price_per_hour": price,
                 "created_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    (out / "instance.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    notify("AutoDL 已抢到 5090", f"{uuid} {info['region_sign']} {price:.2f} 元/小时；登录信息见 outputs/autodl/instance.json",
           args.notify, logf)


if __name__ == "__main__":
    main()
