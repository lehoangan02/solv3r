V=~/LHAN/vggt_omega_test/runs; T=~/LHAN/solv3r_test
until grep -q "peak allocated" $V/ytRhg_kf617_windowed/run.log || grep -q Traceback $V/ytRhg_kf617_windowed/run.log; do sleep 10; done
source ~/miniconda3/etc/profile.d/conda.sh; conda activate solv3r; cd $T/solv3r
python $T/export_ply.py $V/ytRhg_kf617_windowed/out/vggt_windowed.npz $T/kf_ytRhg/kf617 $V/ytRhg_kf617_windowed/out/vggt_windowed.ply > $V/ytRhg_kf617_windowed/export.log 2>&1
python $T/export_ply.py $V/ytRhg_kf100/out/vggt.npz $T/kf_ytRhg/kf100 $V/ytRhg_kf100/out/vggt_filtered.ply --w2c > $V/ytRhg_kf100/export.log 2>&1
cd $T; python cmp4.py cmp/ytRhg_3way_multiview.png kf_ytRhg/all \
 "Solv3R 617 kf,runs/ytRhg_kf617_adaptive001_sky/out/solve.npz,runs/ytRhg_kf617_adaptive001_sky/out/solve.ply,kf_ytRhg/kf617,c2w" \
 "VGGT-Omega windowed 617 kf,$V/ytRhg_kf617_windowed/out/vggt_windowed.npz,$V/ytRhg_kf617_windowed/out/vggt_windowed.ply,kf_ytRhg/kf617,c2w" \
 "VGGT-Omega single 100 kf,$V/ytRhg_kf100/out/vggt.npz,$V/ytRhg_kf100/out/vggt_filtered.ply,kf_ytRhg/kf100,w2c" > cmp/ytRhg_3way.log 2>&1
echo done > cmp/ytRhg_3way.done
