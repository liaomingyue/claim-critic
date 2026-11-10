# 用法：source scripts/remote/bg_sync.sh（需要先设 A=/root/autodl-tmp）
# 保存时本地盘、共享盘都要存，但共享盘那份放后台——训练 / 试采只读写本地盘，不等共享盘。
# bg_sync 源 目标目录：后台 rsync 一次（不挡后续步骤）；日志 $A/bg_sync.log
bg_sync() {
  mkdir -p "$2"
  ( rsync -a "$1" "$2" && echo "$(date '+%F %T') synced $1 -> $2" || echo "$(date '+%F %T') SYNC FAILED $1 -> $2" ) >> ${A:-/root/autodl-tmp}/bg_sync.log 2>&1 &
}
# mirror_loop 进程号 本地目录 共享盘目录：进程活着时每 5 分钟同步一次（不含每步推给 vLLM 的 lora/），进程结束后再同步一次
mirror_loop() {
  local pid=$1 src=$2 dst=$3
  mkdir -p "$dst"
  ( while kill -0 $pid 2>/dev/null; do rsync -a --exclude lora "$src/" "$dst/"; sleep 300; done
    rsync -a --exclude lora "$src/" "$dst/" && echo "$(date '+%F %T') synced $src -> $dst" || echo "$(date '+%F %T') SYNC FAILED $src -> $dst" ) >> ${A:-/root/autodl-tmp}/bg_sync.log 2>&1 &
}
