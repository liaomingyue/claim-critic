#!/bin/bash
# 本地：把 τ²-bench 打成精简包（src + pyproject + retail 数据 + 英文语言包，约 34 MB），供 scripts/remote/setup_tau2.sh 在共享盘上安装。
#   bash scripts/cs_pack_tau2.sh [τ²-bench 仓库目录] [输出 tar]
# 仓库默认在 external_data/agent/tau2-bench（不进 git）；记下提交号，复现时用同一版。
set -eu
T=${1:-external_data/agent/tau2-bench}; OUT=${2:-outputs/cs/tau2_slim.tar}
mkdir -p "$(dirname $OUT)"
ST=$(mktemp -d)
mkdir -p $ST/data/tau2/domains $ST/data/tau2/multilingual
cp -r $T/src $T/pyproject.toml $T/README.md $T/LICENSE $ST/
cp -r $T/data/tau2/domains/retail $ST/data/tau2/domains/
cp -r $T/data/tau2/multilingual/en $T/data/tau2/multilingual/migration_manifest.json $ST/data/tau2/multilingual/
git -C $T log -1 --format='%H %cd' > $ST/TAU2_COMMIT
find $ST -name __pycache__ -prune -exec rm -rf {} +
tar -cf $OUT -C $ST .
rm -rf $ST
echo "packed $OUT ($(du -h $OUT | cut -f1)), tau2 $(git -C $T log -1 --format=%h)"
