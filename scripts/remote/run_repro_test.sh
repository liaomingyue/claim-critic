#!/bin/bash
# 在有 GPU 的服务器上从头复现（实测 scripts/repro 能不能用）。两台机器各一组：
#   repro_bf：打发布包（只打一次，放共享盘）→ 全新克隆代码 → setup → R1 重新打分 → R2 新 critic 组（rp_bf）
#   repro_r1：等发布包 → 全新克隆 → setup → R2 对照组（rp_r1）
#   后完成的那台把另一组的生成和 9B 打分拷过来，跑 R2 比较。结果和日志放 $FS/repro/results/。
# 干净目录只用：发布包（数据 + LoRA）、基座（--base_src 从共享盘拷，等同于从 HF 下载的同版本）、git bundle 里的代码。
set -u
ARM=${1:?rp_bf|rp_r1}
FS=/root/autodl-fs; A=/root/autodl-tmp; REL=$FS/release_v1; OUT=$FS/repro/results; LOG=$A/repro_test_$ARM.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
mkdir -p $OUT $FS/runs/claims
if mkdir $FS/runs/claims/release_pack_v1 2>/dev/null; then
  log "pack release -> $REL"
  (cd $FS/code/critic_model && python -m scripts.release_pack --out $REL > $A/release_pack.log 2>&1) && touch $REL/PACK_DONE || { log "pack FAILED"; tail -5 $A/release_pack.log; exit 1; }
  grep -a "MISSING" $A/release_pack.log | tee -a $LOG
fi
for i in $(seq 1 180); do [ -f $REL/PACK_DONE ] && break; sleep 60; done
[ -f $REL/PACK_DONE ] || { log "release never packed"; exit 1; }
for i in $(seq 1 120); do [ -f $FS/repro/repo.bundle ] && break; sleep 30; done
log "clean clone"
[ -d $A/clean ] || git clone -q -b loop-v1 $FS/repro/repo.bundle $A/clean || { log "clone FAILED"; exit 1; }
cd $A/clean
export FS_ROOT=$A/repro_store TMP_ROOT=$A/repro_work
bash scripts/repro/setup.sh --local $REL --base_src $FS/models >> $LOG 2>&1 || { log "setup FAILED"; tail -20 $LOG; exit 1; }
mkdir -p $OUT/$ARM
if [ $ARM = rp_bf ]; then
  bash scripts/repro/r1_rescore.sh >> $LOG 2>&1; cp $TMP_ROOT/r1_report.txt $TMP_ROOT/r1_cap3.txt $OUT/ 2>/dev/null
  log "R1: $(tail -1 $TMP_ROOT/r1_report.txt 2>/dev/null)"
fi
bash scripts/repro/r2_rag.sh $ARM >> $LOG 2>&1 || { log "R2 $ARM FAILED"; cp $TMP_ROOT/*.log $OUT/$ARM/ 2>/dev/null; exit 1; }
rsync -a outputs/rt/core/loop2_$ARM/eval $OUT/$ARM/ && touch $OUT/$ARM/DONE
OTHER=rp_r1; [ $ARM = rp_r1 ] && OTHER=rp_bf
if mkdir $FS/runs/claims/repro_cmp 2>/dev/null; then   # 两台都完成后只比一次：先完成的拿不到另一组，留给后完成的
  if [ -f $OUT/$OTHER/DONE ]; then
    mkdir -p outputs/rt/core/loop2_$OTHER && rsync -a $OUT/$OTHER/eval outputs/rt/core/loop2_$OTHER/
    bash scripts/repro/r2_rag.sh cmp >> $LOG 2>&1; cp $TMP_ROOT/r2_report.txt $OUT/ 2>/dev/null
    log "R2: $(tail -1 $TMP_ROOT/r2_report.txt 2>/dev/null)"
  else rmdir $FS/runs/claims/repro_cmp; log "other arm not done yet; it will run cmp"; fi
fi
cp $TMP_ROOT/repro.log $OUT/$ARM/ 2>/dev/null; cp $LOG $OUT/$ARM/
log "REPRO_TEST_DONE $ARM"
