source ~/miniconda3/etc/profile.d/conda.sh; conda activate solv3r
T=~/LHAN/solv3r_test; R=$T/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $T/solv3r
for v in RhgjTgcrE5I 5rczkctaJVA; do
  tag=$([ $v = RhgjTgcrE5I ] && echo ytRhg || echo yt5r)
  F=$T/kf_$tag; mkdir -p $F/all
  [ -f $F/all/done ] || { ffmpeg -loglevel error -i $T/yt_$v/video.mp4 -vf scale=960:-2 -q:v 2 $F/all/%05d.jpg && touch $F/all/done; }
  for n in 617 300; do [ -d $F/kf$n ] || python $T/kf.py $F/all $F/kf$n $n > $F/kf$n.log 2>&1; done
  run() { name=$1; script=$2; shift 2; mkdir -p $R/$name; python $T/cap15.py examples/$script --out $R/$name/out "$@" > $R/$name/run.log 2>&1; }
  run ${tag}_kf617_adaptive001 offline.py --images $F/kf617 --adaptive 0.01 --max-calls 64
  run ${tag}_kf300_adaptive001 offline.py --images $F/kf300 --adaptive 0.01 --max-calls 64
  run ${tag}_kf617_incremental incremental.py --images $F/kf617 --initial 8 --group 10 --calls 12
done
echo ALLDONE > $R/kf_chain.done
