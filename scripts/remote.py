"""在 AutoDL 实例上执行命令、上传文件（paramiko，登录信息读 outputs/autodl/web_instance.json，不打印密码）。

用法：
  python -m scripts.remote run "nvidia-smi"                 # 在登录 shell（bash -lc）里执行，conda 环境可用
  python -m scripts.remote put local_path remote_path       # 上传单个文件
  python -m scripts.remote run --bg "cmd" --log /root/autodl-tmp/x.log   # 后台执行（nohup），立即返回
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys

import paramiko

INFO = os.environ.get("AUTODL_INFO", "outputs/autodl/web_instance.json")   # 多台机器时用 AUTODL_INFO 指定登录信息文件


def connect() -> paramiko.SSHClient:
    info = json.load(open(INFO, encoding="utf-8"))
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(info["proxy_host"], port=info["ssh_port"], username="root", password=info["root_password"], timeout=30)
    return c


def run(cmd: str, timeout: int = 3600, pty: bool = True) -> int:
    c = connect()
    _, o, e = c.exec_command(f"bash -lc {shlex.quote(cmd)}", timeout=timeout, get_pty=pty)
    for line in iter(o.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()
    code = o.channel.recv_exit_status()
    c.close()
    return code


def put(local: str, remote: str) -> None:
    """先经 SFTP 传到 /root（AutoDL 的 SFTP 写数据盘路径报 No such file），再 mv 到目标位置。"""
    c = connect()
    tmp = "/root/" + remote.rsplit("/", 1)[-1] + ".part"
    sftp = c.open_sftp()
    sftp.put(local, tmp)
    sftp.close()
    _, o, e = c.exec_command(f"mv {shlex.quote(tmp)} {shlex.quote(remote)}")
    if o.channel.recv_exit_status() != 0:
        raise RuntimeError(e.read().decode())
    c.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=["run", "put"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--bg", action="store_true")
    ap.add_argument("--log", default="/root/autodl-tmp/bg.log")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    if a.op == "put":
        put(a.args[0], a.args[1])
        return
    cmd = " ".join(a.args)
    if a.bg:  # setsid + 不分配 pty、不再套一层 bash -lc：SSH 会话关闭后进程不被一起杀掉
        c = connect()
        _, o, _ = c.exec_command(f"setsid nohup bash -lc {shlex.quote(cmd)} > {a.log} 2>&1 < /dev/null & echo started")
        print(o.read().decode().strip())
        c.close()
        return
    sys.exit(run(cmd, a.timeout))


if __name__ == "__main__":
    main()
