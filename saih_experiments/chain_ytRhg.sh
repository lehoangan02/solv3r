mkdir -p ~/LHAN/solv3r_test/yt_RhgjTgcrE5I && cd ~/LHAN/solv3r_test/yt_RhgjTgcrE5I
~/LHAN/ytdlp_venv/bin/yt-dlp -f 'bv*[height<=1080]+ba/b[height<=1080]' --merge-output-format mp4 -o 'video.%(ext)s' 'https://www.youtube.com/watch?v=RhgjTgcrE5I' > dl.log 2>&1
until [ -f ~/LHAN/solv3r_test/runs/yt5r_chain.done ]; do sleep 10; done
source ~/miniconda3/etc/profile.d/conda.sh; conda activate solv3r
cd ~/LHAN/solv3r_test/solv3r
V=~/LHAN/solv3r_test/yt_RhgjTgcrE5I/video.mp4
R=~/LHAN/solv3r_test/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for cfg in "ytRhg_offline_15gb_16calls:" "ytRhg_offline_15gb_adaptive:--adaptive 0.02"; do
  n=${cfg%%:*}; x=${cfg#*:}; mkdir -p $R/$n
  python ../cap15.py examples/offline.py --images $V --out $R/$n/out --vram-gb 15 $x > $R/$n/run.log 2>&1
done
echo ALLDONE > $R/ytRhg_chain.done
