source ~/miniconda3/etc/profile.d/conda.sh; conda activate solv3r
cd ~/LHAN/solv3r_test/solv3r
V=~/LHAN/solv3r_test/yt_5rczkctaJVA/video.mp4
R=~/LHAN/solv3r_test/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for cfg in "yt5r_offline_15gb_16calls:" "yt5r_offline_15gb_adaptive:--adaptive 0.02"; do
  n=${cfg%%:*}; x=${cfg#*:}; mkdir -p $R/$n
  python ../cap15.py examples/offline.py --images $V --out $R/$n/out --vram-gb 15 $x > $R/$n/run.log 2>&1
done
echo ALLDONE > $R/yt5r_chain.done
