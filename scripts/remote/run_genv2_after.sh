#!/bin/bash
# 生成器输出顺序改成先 sentence 后 edit（edit 描述已写好的句子，不当计划）。等 sev3 / rm2v2 流程做完，归档旧版 gen-v2，重训 2B / 9B 并采样。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; C=outputs/rt/core
until grep -qE 'SEV3_RM2V2_DONE|FAILED' $A/sev3_rm2v2.log 2>/dev/null; do sleep 30; done
for x in gen-v2-mc gen-v2-9b; do [ -d $FS/adapters/$x ] && mv $FS/adapters/$x $FS/archive/$x-editfirst; done
for f in samples_v2_mc samples_v2_9b; do [ -f $C/$f.jsonl ] && mv $C/$f.jsonl $C/$f.editfirst.jsonl; done
rm -rf $A/ckpt/gen-v2-mc $A/ckpt/gen-v2-9b
echo '--- relaunch sentence-first (idx, sentence, edit)' >> $A/genv2.log
bash scripts/remote/run_genv2.sh
