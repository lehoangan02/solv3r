"""Same export as Solv3R's solve.ply (solv3r.export.scene_points: depth-edge pixels, stray far points, sky and low confidence removed) for any c2w/K/depth/conf npz.
usage: export_ply.py NPZ IMAGE_DIR OUT.ply [--w2c] [--keep 0.5] [--no-sky]"""
import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.expanduser("~/LHAN/solv3r_test/solv3r"))
from solv3r.export import scene_points, write_ply
npz, imdir, out = sys.argv[1:4]; w2c = "--w2c" in sys.argv; sky = "--no-sky" not in sys.argv
keep = float(sys.argv[sys.argv.index("--keep") + 1]) if "--keep" in sys.argv else 0.5
Z = np.load(npz); d = Z["depth"].astype(np.float32); c = Z["depth_conf"].astype(np.float32); names = Z["names"]; S, H, W = d.shape
if w2c:
    E = Z["extrinsics"].astype(np.float64); K = Z["intrinsics"]; Rc = E[:, :, :3].transpose(0, 2, 1); cen = -(Rc @ E[:, :, 3][..., None])[..., 0]
else:
    K = Z["K"]; Rc = Z["c2w"][:, :3, :3].astype(np.float64); cen = Z["c2w"][:, :3, 3].astype(np.float64)
v, u = np.mgrid[0:H, 0:W]
x = (u[None] - K[:, 0, 2][:, None, None]) / K[:, 0, 0][:, None, None] * d; y = (v[None] - K[:, 1, 2][:, None, None]) / K[:, 1, 1][:, None, None] * d
pts = (np.einsum("sij,shwj->shwi", Rc, np.stack([x, y, d], -1)) + cen[:, None, None, :]).astype(np.float32)
im = np.stack([cv2.resize(cv2.imread(os.path.join(imdir, str(nm))), (W, H))[:, :, ::-1] for nm in names]).transpose(0, 3, 1, 2) / 255.0
xyz, rgb = scene_points(pts, c, d, im.astype(np.float32), keep, sky)
write_ply(out, xyz, rgb); print("wrote", len(xyz), "points ->", out)
