#!/bin/bash
# 生成器输出顺序修改优先级最高 → 先重训生成器（run_genv2_after 的内容），做完再跑 sev3 / rm2v2（描述和数据已就绪的步骤会跳过）。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; C=outputs/rt/core
for x in gen-v2-mc gen-v2-9b; do [ -d $FS/adapters/$x ] && mv $FS/adapters/$x $FS/archive/$x-editfirst; done
for f in samples_v2_mc samples_v2_9b; do [ -f $C/$f.jsonl ] && mv $C/$f.jsonl $C/$f.editfirst.jsonl; done
rm -rf $A/ckpt/gen-v2-mc $A/ckpt/gen-v2-9b
echo '--- relaunch sentence-first (idx, sentence, edit)' >> $A/genv2.log
bash scripts/remote/run_genv2.sh
echo '--- resume after generator retrain' >> $A/sev3_rm2v2.log
bash scripts/remote/run_sev3_rm2v2.sh
