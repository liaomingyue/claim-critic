#!/bin/bash
# 等本机 run_worker2.sh 结束（KL 触发后它会标 FAILED 退出），再用放宽的 KL 上限续跑同一档。用法：ARM=... bash chase2.sh
while pgrep -f 'run_worker2.sh' > /dev/null; do touch /root/autodl-tmp/KEEPALIVE; sleep 30; done
ARM=$ARM bash /root/autodl-fs/code/critic_model/scripts/remote/run_r200c.sh
