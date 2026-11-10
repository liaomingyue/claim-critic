#!/bin/bash
# 本地盘模型副本（暂时用不到的先不拷，先拷最需要的，9B 这种后台新开线程拷，把时间岔开）。
# source 进脚本后：
#   start_9b_bg  脚本开头调用：后台把 Qwen3.5-9B 拷到本地盘（低 IO 优先级），不挡当前步骤
#   ensure_9b    真要用 9B 时调用：已拷完直接返回；后台还在拷就等它（同一把锁），没开始就前台拷
# 先拷进临时目录，拷完才改名为正式目录（别的脚本只看正式目录的 config.json，不会读到拷了一半的权重）；flock 保证一台机器只有一个拷贝。
M9_SRC=${FS_ROOT:-/root/autodl-fs}/models/Qwen3.5-9B; M9_DIR=${TMP_ROOT:-/root/autodl-tmp}/models; M9=$M9_DIR/Qwen3.5-9B; M9_LOCK=${TMP_ROOT:-/root/autodl-tmp}/.copy9b.lock
_copy_9b() {
  mkdir -p $M9_DIR
  (
    flock 9
    [ -f $M9/config.json ] && exit 0
    t0=$(date +%s)
    ionice -c2 -n7 nice -n 10 rsync -a $M9_SRC/ $M9_DIR/.Qwen3.5-9B.partial/ || exit 1
    if [ -e $M9 ]; then rm -rf $M9_DIR/.Qwen3.5-9B.partial; else mv $M9_DIR/.Qwen3.5-9B.partial $M9; fi
    echo "$(date '+%F %T') copy9b done in $(( $(date +%s) - t0 ))s" >> ${TMP_ROOT:-/root/autodl-tmp}/copy9b.log
  ) 9>$M9_LOCK
}
start_9b_bg() { [ -f $M9/config.json ] || ( _copy_9b > /dev/null 2>&1 & ); }
ensure_9b() { [ -f $M9/config.json ] || _copy_9b; [ -f $M9/config.json ]; }
