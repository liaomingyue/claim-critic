"""把本地改动的小文件写进共享盘的代码目录（经一台开着的实例；走 SSH 命令通道分块写，不用 SFTP）。
  python -m scripts.fs_push --via <实例信息 json> <文件或目录> ...
只适合小文件（代码、标注）；大文件请在远程生成。"""
import argparse, base64, hashlib, json
from pathlib import Path
from scripts.autodl_extra import sh

R = "/root/autodl-fs/code/critic_model"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--via", required=True)
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    info = json.load(open(a.via))
    files = [f for p in a.paths for f in ([Path(p)] if Path(p).is_file() else sorted(x for x in Path(p).rglob("*") if x.is_file()))]
    for f in files:
        b = f.read_bytes(); d = f"{R}/{f.as_posix()}"; tmp = d + ".part"
        sh(info, f"mkdir -p $(dirname {d}) && : > {tmp}")
        enc = base64.b64encode(b).decode()
        for i in range(0, len(enc), 60000):
            sh(info, f"echo '{enc[i:i + 60000]}' | base64 -d >> {tmp}")
        got = sh(info, f"sha256sum {tmp} | cut -c1-16").strip()
        assert got == hashlib.sha256(b).hexdigest()[:16], (f, got)
        sh(info, f"mv {tmp} {d}")
        print("pushed", f, len(b))


if __name__ == "__main__":
    main()
