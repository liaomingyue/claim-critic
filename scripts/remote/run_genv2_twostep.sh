#!/bin/bash
# 2026-11-08 用户：错误生成器两步生成——第一步只出 idx + sentence；第二步同一模型看原句 / 改写句 / 逐词改动写 edit（给 sev 用）。
# 多任务 SFT（改写 1394 + 描述 1394）。等同机的 sev3 / rm2v2 打分做完，归档上一版 gen-v2，重训 2B / 9B 并两步采样。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; C=outputs/rt/core
until grep -qE 'SEV3_RM2V2_DONE' $A/sev3_rm2v2.log 2>/dev/null; do sleep 30; done
for x in gen-v2-mc gen-v2-9b; do [ -d $FS/adapters/$x ] && mv $FS/adapters/$x $FS/archive/$x-sentfirst; done
for f in samples_v2_mc samples_v2_9b; do [ -f $C/$f.jsonl ] && mv $C/$f.jsonl $C/$f.sentfirst.jsonl; done
rm -rf $A/ckpt/gen-v2-mc $A/ckpt/gen-v2-9b
echo '--- relaunch two-step (rewrite + describe multitask)' >> $A/genv2.log
bash scripts/remote/run_genv2.sh
