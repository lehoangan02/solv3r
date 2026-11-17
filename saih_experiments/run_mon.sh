#!/bin/bash
# usage: run_mon.sh NAME VIDEO [extra args]
NAME=$1; VID=$2; shift 2
source ~/miniconda3/etc/profile.d/conda.sh; conda activate solv3r
D=~/LHAN/solv3r_test/runs/$NAME; mkdir -p $D
cd ~/LHAN/solv3r_test/solv3r
( while true; do echo "$(date +%s) $(free -m | awk "/Mem:/{print \$3}") $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d " ")"; sleep 2; done ) > $D/mon.log &
MP=$!
S=$(date +%s); python examples/streaming.py --frames $VID --out $D/out "$@" > $D/run.log 2>&1; echo wall_seconds=$(($(date +%s)-S)) >> $D/run.log
kill $MP
echo done >> $D/run.log
