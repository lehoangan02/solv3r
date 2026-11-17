T=~/LHAN/solv3r_test; R=$T/runs
until [ -f $R/kf_chain.done ]; do sleep 10; done
source ~/miniconda3/etc/profile.d/conda.sh; conda activate solv3r
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
n=bedroom1_L8_ms4_keep05; mkdir -p $R/$n; cd $T/solv3r
python $T/cap15.py examples/streaming.py --frames ~/LHAN/RoSE/Bedroom1/ScanVideos/Video1.mp4 --out $R/$n/out --config L8 --memory-stride 4 --keep 0.5 > $R/$n/run.log 2>&1
echo done > $R/keep05.done
