"""盯 5090 上的实验流水线：完成（exp.log 出现 ALL_DONE）、流水线进程没了、或 GPU 长时间空闲时，
下载评测结果、关机并通知。关机前先把结果拉回本地，数据盘上的内容关机后也保留。

用法：python -m scripts.autodl_watch --uuid 611b489072-8df08a9e
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from scripts.autodl_grab import log, notify
from scripts.autodl_web import Web
from scripts.remote import connect

REMOTE = "/root/autodl-tmp/critic_model"


def sh(cmd: str) -> str:
    c = connect()
    _, o, _ = c.exec_command(cmd)
    out = o.read().decode(errors="replace")
    c.close()
    return out


def safe_get(s, remote: str, local: str) -> bool:
    """先下到 .part 再替换；本地已有且远端更小（加开机器上只有不完整的副本）时不覆盖。
    paramiko 的 get 会先清空本地文件再读远端，直接 get 曾把本地完整的 18,919 行预测截成几千行（2026-10-27）。"""
    try:
        size = s.stat(remote).st_size
    except OSError:
        return False
    if os.path.exists(local) and os.path.getsize(local) > size:
        return False
    s.get(remote, local + ".part")
    os.replace(local + ".part", local)
    return True


def download_evals(logf: Path, log_dir: str = "outputs/autodl/remote_logs") -> int:
    c = connect()
    s = c.open_sftp()
    n = 0
    for d in s.listdir(f"{REMOTE}/outputs/rt/eval"):
        os.makedirs(f"outputs/rt/eval/{d}", exist_ok=True)
        for f in ("metrics.json", "predictions.jsonl"):
            n += safe_get(s, f"{REMOTE}/outputs/rt/eval/{d}/{f}", f"outputs/rt/eval/{d}/{f}")
    try:   # 合成样本、识破率、三组训练集的统计（ragtruth.synth / ragtruth.mine）
        os.makedirs("outputs/rt/synth", exist_ok=True)
        for f in s.listdir(f"{REMOTE}/outputs/rt/synth"):
            safe_get(s, f"{REMOTE}/outputs/rt/synth/{f}", f"outputs/rt/synth/{f}")
        for f in s.listdir(f"{REMOTE}/outputs/rt/claims"):
            if f.startswith("syn_") and f.endswith(".meta.json"):
                safe_get(s, f"{REMOTE}/outputs/rt/claims/{f}", f"outputs/rt/claims/{f}")
    except OSError:
        pass
    try:   # 负例复核结果（ragtruth.negcheck），两台机器各写自己那一片
        os.makedirs("outputs/rt/negcheck", exist_ok=True)
        for f in s.listdir(f"{REMOTE}/outputs/rt/negcheck"):
            if f.endswith(".jsonl"):
                safe_get(s, f"{REMOTE}/outputs/rt/negcheck/{f}", f"outputs/rt/negcheck/{f}")
    except OSError:
        pass
    os.makedirs(log_dir, exist_ok=True)
    for f in s.listdir("/root/autodl-tmp"):   # 训练 / 评测日志一并拉回，画 loss 曲线和排查用
        if f.endswith(".log"):
            s.get(f"/root/autodl-tmp/{f}", f"{log_dir}/{f}")
    c.close()
    log(f"downloaded {n} eval files", logf)
    return n


def main() -> None:
    global REMOTE
    ap = argparse.ArgumentParser()
    ap.add_argument("--uuid", default=None, help="默认读 outputs/autodl/web_instance.json（克隆后 uuid 会变）")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--idle_minutes", type=int, default=20, help="流水线不在且 GPU 空闲这么久就收尾")
    ap.add_argument("--notify", nargs="*", default=["email", "toast"])
    ap.add_argument("--exp_log", default="/root/autodl-tmp/exp.log", help="流水线日志，最后一行出现 --done 即完成")
    ap.add_argument("--done", default="ALL_DONE")
    ap.add_argument("--proc", default="run_format_exp", help="流水线脚本名，用来判断进程是否还在")
    ap.add_argument("--keep_on_file", default="outputs/autodl/keep_on", help="这个文件存在就不关机；none = 总是关机（加开的机器）")
    ap.add_argument("--log_dir", default="outputs/autodl/remote_logs", help="远端日志下载到哪（多台机器各用一个目录）")
    ap.add_argument("--gpu_idle_only", type=int, default=1, help="1 = 只看 GPU：所有卡利用率 ≤ 5% 持续 idle_minutes 就收尾，流水线进程还在也关（卡死时不空烧，用户 2026-11-03）")
    ap.add_argument("--remote_dir", default=REMOTE, help="远端代码目录；文件存储模式为 /root/autodl-fs/code/critic_model")
    args = ap.parse_args()

    logf = Path("outputs/autodl/watch.log")
    if args.uuid is None:
        args.uuid = json.loads(Path("outputs/autodl/web_instance.json").read_text(encoding="utf-8"))["uuid"]
    idle_since = None
    reason = ""
    while True:
        try:
            status = sh(f"tail -1 {args.exp_log}; echo '|'; ps aux | grep -c '[{args.proc[0]}]{args.proc[1:]}'; echo '|';"
                        " nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits")
            last, running, util = [x.strip() for x in status.split("|")]
        except Exception as e:  # noqa: BLE001 网络抖动，下次再看
            log(f"ssh failed: {e}", logf)
            time.sleep(args.interval)
            continue
        if args.done in last:
            reason = "实验全部完成"
            break
        utils = [int(u) for u in util.split() if u.isdigit()]   # 多卡时每张卡一行
        gpu_busy = any(u > 5 for u in utils)
        busy = (running != "0" or gpu_busy) if not args.gpu_idle_only else gpu_busy
        idle_since = None if busy else (idle_since or time.time())
        ov = Path("outputs/autodl/idle_minutes")   # 运行中改空闲时长（用户：GPU 空闲 20 分钟自动关机）
        idle_min = int(ov.read_text().strip()) if ov.exists() else args.idle_minutes
        if idle_since and time.time() - idle_since > idle_min * 60:
            reason = f"GPU 空闲 {idle_min} 分钟（流水线进程 {running} 个，最后一步：{last}）"
            break
        time.sleep(args.interval)

    log(f"finishing: {reason}", logf)
    REMOTE = args.remote_dir
    try:   # 下载失败（目录不在等）不能挡住关机：2026-10-28 文件存储模式下这里抛错，机器没关、空转
        download_evals(logf, args.log_dir)
    except Exception as e:  # noqa: BLE001
        log(f"download failed ({type(e).__name__}: {e}); powering off anyway", logf)
    if args.keep_on_file != "none" and Path(args.keep_on_file).exists():  # 后面还要接着做实验：只下载结果，不关机
        log("keep_on: skip power_off", logf)
        notify("AutoDL 实验结束，保持开机", f"{reason}；结果已下载到本地 outputs/rt/eval；按要求未关机", args.notify, logf)
        return
    web = Web("autodl_web.json")
    r = web.call("/api/v1/instance/power_off", {"instance_uuid": args.uuid})
    log(f"power_off: {r.get('code')} {r.get('msg')}", logf)
    notify("AutoDL 实验结束，已关机", f"{reason}；结果已下载到本地 outputs/rt/eval；关机：{r.get('code')}", args.notify, logf)


if __name__ == "__main__":
    main()
