"""AutoDL 抢卡（网页接口版）：用你在自己浏览器里正常登录后的会话，轮询算力市场，有空闲 5090 就创建普通实例（按量计费）。

和 autodl_grab.py 的区别：官方 Pro API 的创建接口对本账号返回「无当前资源访问权限」，这里改用网页控制台
自己调用的接口。这些接口不是官方公开 API（路径和字段参考 github.com/Fo-OLLL/autodl-card-helper），
网页改版时可能失效；使用前请自行确认符合 AutoDL 使用条款。脚本不碰登录和验证码，只用你登录后的会话。

会话：在已登录的 autodl.com 页面按 F12 → Console，运行
  copy(JSON.stringify({token: localStorage.getItem('token'), app_version: JSON.parse(localStorage.getItem('app_version') || '{}').version}))
把剪贴板内容粘到 autodl_web.json（本地，不进仓库）。会话过期后重新复制一次。

流程：
1. 每 interval 秒查一次算力市场：西北B区（westDC3，网页端分区 west-B/C/D/E）、RTX 5090、至少 1 张空闲、按量计费；
   排除离线/维护中的主机（on_line 3–6）和已绑满实例的主机，数据盘可扩容空间不够的也排除。
2. 有合适主机就下单创建：1 张卡、基础镜像（PyTorch，CUDA ≥ 12.8）、数据盘扩容 expand_gb；
   实例名 ADLH-<时间戳>，超时或返回异常时先查实例列表，不重复下单。
3. 同一时间最多 1 台：实例列表里已有本脚本创建且未释放的实例就不再创建。
4. 创建成功后等实例 running，记录到 outputs/autodl/web_instance.json，并发邮件 / 弹窗通知（不含密码）。

用法：
  python -m scripts.autodl_web --dry_run      # 只读：检查会话、列出镜像候选和当前空闲主机，不下单
  python -m scripts.autodl_web --expand_gb 150
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from scripts.autodl_grab import log, notify

BASE = "https://www.autodl.com"
GIB = 1024 ** 3
OFFLINE = {3, 4, 5, 6}
GONE = {"released", "release", "create_failed"}


class Web:
    def __init__(self, session_file: str):
        s = json.loads(Path(session_file).read_text(encoding="utf-8"))
        self.headers = {"Authorization": s["token"], "AppVersion": s.get("app_version") or "0.0.0",
                        "Content-Type": "application/json", "Referer": BASE + "/console/instance/list"}

    def call(self, path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(BASE + path, method="POST" if body is not None else "GET",
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return {"code": f"HTTP{e.code}", "msg": e.read()[:300].decode("utf-8", "replace")}
        except (urllib.error.URLError, TimeoutError) as e:
            return {"code": "NetworkError", "msg": str(e)}

    def region_signs(self, data_center: str) -> list[str]:
        """网页端一个地区对应多个分区标识（西北B区 westDC3 → west-B / west-C / west-D / west-E）。"""
        r = self.call("/api/v1/region/list")
        return [x["region_sign"] for x in r.get("data") or []
                if x.get("data_center") == data_center and x.get("visible") == "all"]

    def machines(self, signs: list[str], gpu: str, idle: int = 1) -> list[dict]:
        """逐页取全（每页最多 90 台；只取第一页时一个地区 90+ 台 5090 会漏掉后面的空闲机器）。"""
        out = []
        for page in range(1, 50):
            r = self.call("/api/v1/user/machine/list", {
                "charge_type": "payg", "gpu_type_name": [gpu], "gpu_idle_num": idle, "region_sign_list": signs,
                "default_order": True, "machine_tag_name": [], "mount_net_disk": False, "cpu_arch": [], "chip_corp": [],
                "instance_disk_size_order": "", "date_range": "", "date_from": "", "date_to": "",
                "pay_price_order": "", "gpu_idle_type": "", "page_index": page, "page_size": 90})
            if r.get("code") != "Success":
                raise RuntimeError(f"machine list failed: {r.get('code')} {r.get('msg')}")
            data = r.get("data") or {}
            rows = data.get("list") or []
            out += rows
            if len(rows) < 90 or len(out) >= (data.get("result_total") or data.get("total") or 0):
                break
        return out

    def instances(self) -> list[dict]:
        r = self.call("/api/v1/instance", {"status": [], "charge_type": [], "name": "", "sub_name": "",
                                           "page_index": 1, "page_size": 90})
        if r.get("code") != "Success":
            raise RuntimeError(f"instance list failed: {r.get('code')} {r.get('msg')}")
        return (r.get("data") or {}).get("list") or []

    def images(self) -> list[dict]:
        """基础镜像树的叶子：[{name, uuid, v}]。"""
        r = self.call("/api/v1/image/all", {"chip_corp": ["nvidia"], "cpu_arch": ["x86"]})
        out = []

        def walk(node, path):
            if isinstance(node, list):
                for n in node:
                    walk(n, path)
            elif isinstance(node, dict):
                label = node.get("label_name")
                name = node.get("label") or node.get("name") or ""
                if isinstance(label, dict) and label.get("uuid"):
                    out.append({"name": " / ".join(path + [name]), "uuid": label["uuid"], "v": label.get("v")})
                for k in ("children", "list"):
                    if k in node:
                        walk(node[k], path + ([name] if name else []))

        walk(r.get("data"), [])
        return out

    def create(self, machine_id: str, image_uuid: str, expand_bytes: int, name: str) -> dict:
        # 自定义镜像（控制台「保存镜像」得到的 image-xxx）放 private_image_uuid；放进 base_image_uuid 会报「基础镜像不存在」
        private = image_uuid.startswith("image-")
        return self.call("/api/v1/order/instance/create/payg", {
            "instance_info": {"machine_id": machine_id, "charge_type": "payg", "req_gpu_amount": 1,
                              "base_image_uuid": "" if private else image_uuid,
                              "private_image_uuid": image_uuid if private else "", "reproduction_uuid": "",
                              "reproduction_id": 0, "cg_application_uuid": "", "instance_name": name,
                              "expand_data_disk": expand_bytes},
            "price_info": {"coupon_id_list": [], "machine_id": machine_id, "charge_type": "payg", "duration": 1,
                           "num": 1, "expand_data_disk": expand_bytes}})


def usable(m: dict, expand_bytes: int, max_price: float = float("inf")) -> bool:
    """有空闲卡、在线可租（on_line 3–6 为离线 / 维护，空闲卡也租不了）、没绑满、数据盘能扩到、
    每卡每小时价格（payg_price，单位 元 × 1000）不超过 max_price。"""
    return (m.get("gpu_idle_num", 0) >= 1 and m.get("on_line") not in OFFLINE
            and m.get("binding_instance_num", 0) < m.get("max_instance_num", 1 << 30)
            and m.get("max_data_disk_expand_size", 1 << 62) >= expand_bytes
            and m.get("payg_price", 0) / 1000 <= max_price)


def pick_image(images: list[dict], keyword: str) -> dict | None:
    """名字形如「PyTorch / 2.8.0 / 3.12(ubuntu22.04) / 12.8」（最后一段是 CUDA 版本）。
    取 CUDA 恰好 12.8 的（5090 需要 ≥ 12.8；vLLM / torch 官方轮子以 cu128 为主），没有再取 ≥ 12.8 里最低的。"""
    def cuda(name: str) -> float:
        try:
            return float(name.rsplit("/", 1)[-1].strip())
        except ValueError:
            return 0.0

    cands = [i for i in images if keyword.lower() in i["name"].lower() and cuda(i["name"]) >= 12.8]
    return min(cands, key=lambda i: cuda(i["name"])) if cands else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="autodl_web.json")
    ap.add_argument("--region", default="westDC3", help="数据中心（西北B区 = westDC3），自动展开成网页端的分区标识")
    ap.add_argument("--gpu", default="RTX 5090")
    ap.add_argument("--image", default=None, help="基础镜像 uuid；不给则自动选 PyTorch + CUDA ≥ 12.8 的最新版")
    ap.add_argument("--image_keyword", default="PyTorch")
    ap.add_argument("--expand_gb", type=int, default=150, help="数据盘扩容（GB），默认 50G 免费盘 + 150G = 200G")
    ap.add_argument("--max_price", type=float, default=5.0, help="每卡每小时最高价（元）")
    ap.add_argument("--interval", type=int, default=60, help="轮询间隔（秒），不要太频繁")
    ap.add_argument("--max_hours", type=float, default=24)
    ap.add_argument("--notify", nargs="*", default=["email", "toast"])
    ap.add_argument("--out_dir", default="outputs/autodl")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logf = out / "web.log"
    web = Web(args.session)
    expand = args.expand_gb * GIB

    mine = [i for i in web.instances() if str(i.get("name", "")).startswith("ADLH-") and i.get("status") not in GONE]
    log(f"session ok; {len(mine)} instance(s) created by this script still exist", logf)
    images = web.images()
    image = {"uuid": args.image, "name": args.image} if args.image else pick_image(images, args.image_keyword)
    signs = web.region_signs(args.region)
    if not signs:
        raise SystemExit(f"网页端没有 {args.region} 的分区")
    if args.dry_run:
        for i in [i for i in images if args.image_keyword.lower() in i["name"].lower()][-15:]:
            log(f"image candidate: {i['name']} {i['uuid']}", logf)
        log(f"selected image: {image}", logf)
        log(f"{args.region} -> {signs}; {len(web.machines(signs, args.gpu, idle=0))} {args.gpu} machine(s) in total", logf)
        ms = web.machines(signs, args.gpu)
        log(f"{len(ms)} machine(s) with idle {args.gpu}; usable {sum(usable(m, expand) for m in ms)}", logf)
        if ms:
            log("first machine fields: " + json.dumps(ms[0], ensure_ascii=False)[:800], logf)
        return
    if image is None:
        raise SystemExit("没找到 PyTorch + CUDA ≥ 12.8 的基础镜像，用 --image 指定")
    if mine:
        raise SystemExit(f"已有本脚本创建的实例 {[i.get('uuid') for i in mine]}，不再创建")

    log(f"polling {args.region} {args.gpu} every {args.interval}s with image {image['name']}", logf)
    deadline, tries = time.time() + args.max_hours * 3600, 0
    while time.time() < deadline:
        tries += 1
        try:
            cands = [m for m in web.machines(signs, args.gpu) if usable(m, expand, args.max_price)]
        except RuntimeError as e:
            log(str(e), logf)
            if "HTTP401" in str(e) or "登录" in str(e):
                notify("AutoDL 会话已过期", "请重新登录并更新 autodl_web.json", args.notify, logf)
                raise SystemExit(3)
            time.sleep(args.interval)
            continue
        for m in cands:
            name = f"ADLH-{time.strftime('%m%d%H%M%S')}"
            r = web.create(m["machine_id"], image["uuid"], expand, name)
            if r.get("code") == "Success":
                log(f"created {name} on {m['machine_id']} ({m.get('region_name')}) after {tries} polls: {r.get('data')}",
                    logf)
                inst = next((i for i in web.instances() if i.get("name") == name), {"name": name})
                (out / "web_instance.json").write_text(json.dumps(inst, ensure_ascii=False, indent=2), encoding="utf-8")
                notify("AutoDL 已抢到 5090", f"{name} {m.get('region_name')} 主机 {m['machine_id']}；"
                       f"登录信息在控制台或 outputs/autodl/web_instance.json", args.notify, logf)
                return
            log(f"create on {m['machine_id']} failed: {r.get('code')} {r.get('msg')}", logf)
            if any(i.get("name") == name for i in web.instances()):   # 返回异常但实际已创建：不重复下单
                notify("AutoDL 已抢到 5090", f"{name}（下单返回异常，但实例已存在，请到控制台确认）", args.notify, logf)
                return
        if tries == 1 or tries % 30 == 0:
            log(f"poll {tries}: no usable {args.gpu} in {args.region}", logf)
        time.sleep(args.interval)
    notify("AutoDL 未抢到 5090", f"{args.max_hours} 小时内没有空闲", args.notify, logf)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
