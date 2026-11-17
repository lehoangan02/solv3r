"""Solv3R (617 keyframes) vs VGGT-Omega (100 keyframes): align by trajectory, render colour point clouds from several viewpoints."""
import sys, os, glob, hashlib, numpy as np, cv2
sol_npz, vg_npz, sol_dir, vg_dir, all_dir, out = sys.argv[1:7]
S = np.load(sol_npz); V = np.load(vg_npz)
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
allf = sorted(glob.glob(all_dir + "/*.jpg")); idx_of = {md5(p): i for i, p in enumerate(allf)}
def orig_idx(d, names): return np.array([idx_of[md5(os.path.join(d, str(n)))] for n in names])
iS = orig_idx(sol_dir, S["names"]); iV = orig_idx(vg_dir, V["names"])
print("original frame index ranges: Solv3R", iS.min(), iS.max(), "(", len(iS), ") VGGT", iV.min(), iV.max(), "(", len(iV), ")", "exact shared frames:", len(set(iS) & set(iV)))
cS_all = S["c2w"][:, :3, 3]
o = np.argsort(iS); iS, cS_all = iS[o], cS_all[o]
cS = np.stack([np.interp(iV, iS, cS_all[:, k]) for k in range(3)], 1)     # Solv3R trajectory interpolated at VGGT's frames
E = V["extrinsics"]; Rc = E[:, :, :3].transpose(0, 2, 1); cV = -(Rc @ E[:, :, 3][..., None])[..., 0]
def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0); X, Y = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(Y.T @ X / len(src)); Sg = np.eye(3); Sg[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ Sg @ Vt; s = (D * Sg.diagonal()).sum() / X.var(0).sum(); return s, R, mu_d - s * R @ mu_s
s, R, t = umeyama(cV, cS); al = (s * (R @ cV.T)).T + t
ext = np.linalg.norm(cS - cS.mean(0), axis=1).max(); rmse = np.sqrt(((al - cS) ** 2).sum(1).mean())
print(f"trajectory agreement: RMSE {rmse:.4f} = {100*rmse/(2*ext):.1f}% of extent (positions of Solv3R interpolated between its keyframes), scale {s:.3f}")
def cloud(img_dir, names, depth, conf, K, c2w, keep=0.5, step=2):
    pts, cols = [], []
    thr = np.quantile(conf[:, ::7, ::7], 1 - keep); H, W = depth.shape[1:]
    v, u = np.mgrid[0:H:step, 0:W:step]
    for i in range(len(depth)):
        im = cv2.resize(cv2.imread(os.path.join(img_dir, str(names[i]))), (W, H))[::step, ::step, ::-1]
        d = depth[i, ::step, ::step]; m = (conf[i, ::step, ::step] >= thr) & (d > 0); z = d[m]
        x = (u[m] - K[i, 0, 2]) / K[i, 0, 0] * z; y = (v[m] - K[i, 1, 2]) / K[i, 1, 1] * z
        pts.append((c2w[i, :3, :3] @ np.stack([x, y, z]) + c2w[i, :3, 3:4]).T); cols.append(im[m])
    return np.concatenate(pts).astype(np.float32), np.concatenate(cols)
n = len(V["names"]); c2wV = np.zeros((n, 4, 4)); c2wV[:, 3, 3] = 1; c2wV[:, :3, :3] = R @ Rc; c2wV[:, :3, 3] = al
PS, CS = cloud(sol_dir, S["names"], S["depth"].astype(np.float32), S["depth_conf"].astype(np.float32), S["K"], S["c2w"])
PV, CV = cloud(vg_dir, V["names"], V["depth"].astype(np.float32) * s, V["depth_conf"].astype(np.float32), V["intrinsics"], c2wV)
print("points: Solv3R", len(PS), " VGGT", len(PV))
# coverage: occupied voxels (1.5% of scene extent) and how much of each the other also fills
allp = np.concatenate([PS, PV]); lo, hi = np.percentile(allp, 1, axis=0), np.percentile(allp, 99, axis=0); ext_scene = (hi - lo).max()
vox = ext_scene * 0.015
def occ(P): 
    m = (P >= lo - 0.05 * ext_scene).all(1) & (P <= hi + 0.05 * ext_scene).all(1)
    return set(map(tuple, np.unique(np.floor(P[m] / vox).astype(np.int32), axis=0)))
oS, oV = occ(PS), occ(PV)
print(f"occupied voxels ({vox:.3f} units): Solv3R {len(oS)}, VGGT {len(oV)}, both {len(oS & oV)}; Solv3R voxels also in VGGT {100*len(oS & oV)/len(oS):.0f}%, VGGT voxels also in Solv3R {100*len(oS & oV)/len(oV):.0f}%")
# views: 3 orthographic (top-down, front, side in Solv3R world = camera 0 frame) + 2 perspective from trajectory cameras
cen = (lo + hi) / 2; SZ = 600
def render_ortho(P, C, a, b, flip_b):
    u = ((P[:, a] - (cen[a] - ext_scene / 2)) / ext_scene * (SZ - 1)).astype(int); w = ((P[:, b] - (cen[b] - ext_scene / 2)) / ext_scene * (SZ - 1)).astype(int)
    if flip_b: w = SZ - 1 - w
    k = (u >= 0) & (u < SZ) & (w >= 0) & (w < SZ); depth = P[k][:, 3 - a - b]; o = np.argsort(-depth)   # far first, near overwrites
    img = np.full((SZ, SZ, 3), 255, np.uint8); img[w[k][o], u[k][o]] = C[k][o][:, ::-1]; return img
def render_persp(P, C, c2w, f=0.9):
    w2c = np.linalg.inv(c2w); X = P @ w2c[:3, :3].T + w2c[:3, 3]; k = X[:, 2] > 0.05 * ext_scene; X, Ck = X[k], C[k]
    u = (f * SZ * X[:, 0] / X[:, 2] + SZ / 2).astype(int); w = (f * SZ * X[:, 1] / X[:, 2] + SZ / 2).astype(int)
    kk = (u >= 0) & (u < SZ) & (w >= 0) & (w < SZ); o = np.argsort(-X[kk][:, 2])
    img = np.full((SZ, SZ, 3), 255, np.uint8); img[w[kk][o], u[kk][o]] = Ck[kk][o][:, ::-1]; return img
def view_cam(i, back=0.5):                                  # a Solv3R camera pose, pulled back along its viewing axis, then used to look at the scene
    c = S["c2w"][i].copy(); c[:3, 3] -= c[:3, 2] * back * ext_scene; return c
order = np.argsort(iS); ids = [order[int(len(order) * q)] for q in (0.25, 0.75)]
# S["c2w"] rows are in file order of S["names"], iS was sorted -> use original indices of the unsorted array
iS_unsorted = orig_idx(sol_dir, S["names"]); ids = [int(np.argsort(iS_unsorted)[int(len(iS_unsorted) * q)]) for q in (0.25, 0.75)]
rows = []
for P, C, lab in ((PS, CS, "Solv3R 617 kf"), (PV, CV, "VGGT-Omega 100 kf")):
    ims = [render_ortho(P, C, 0, 2, True), render_ortho(P, C, 0, 1, False), render_ortho(P, C, 2, 1, False)] + [render_persp(P, C, view_cam(i)) for i in ids]
    for im, t_ in zip(ims, ["top-down", "front", "side", "persp 25% of path", "persp 75% of path"]): cv2.putText(im, t_, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
    cv2.putText(ims[0], lab, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 0, 0), 2); rows.append(np.hstack(ims))
cv2.imwrite(out + "_multiview.png", np.vstack(rows)); print("wrote", out + "_multiview.png")
