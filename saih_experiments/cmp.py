import sys, numpy as np, cv2
sol_npz, vg_npz, out = sys.argv[1], sys.argv[2], sys.argv[3]
S = np.load(sol_npz); V = np.load(vg_npz)
n = len(S["c2w"]); assert (S["names"] == V["names"]).all()
def umeyama(src, dst):                                  # dst ~ s R src + t
    mu_s, mu_d = src.mean(0), dst.mean(0); X, Y = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(Y.T @ X / len(src)); Sg = np.eye(3); Sg[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ Sg @ Vt; s = (D * Sg.diagonal()).sum() / X.var(0).sum(); t = mu_d - s * R @ mu_s
    return s, R, t
cS = S["c2w"][:, :3, 3]
res = {}
for name, conv in (("w2c", "w2c"), ("c2w", "c2w")):
    E = V["extrinsics"]; R_ = E[:, :, :3]; T_ = E[:, :, 3]
    if conv == "w2c": Rc = R_.transpose(0, 2, 1); cV = -(Rc @ T_[..., None])[..., 0]     # camera->world rotation, centre
    else: Rc = R_; cV = T_
    s, R, t = umeyama(cV, cS); al = (s * (R @ cV.T)).T + t
    rmse = np.sqrt(((al - cS) ** 2).sum(1).mean()); ext = np.linalg.norm(cS - cS.mean(0), axis=1).max()
    ang = [np.degrees(np.arccos(np.clip((np.trace(R @ Rc[i] @ S["c2w"][i, :3, :3].T) - 1) / 2, -1, 1))) for i in range(n)]
    res[name] = (rmse / (2 * ext), np.median(ang), s, R, t, Rc, cV)
    print(f"VGGT extrinsics read as {name}: camera-centre RMSE after Sim3 = {rmse:.4f} ({100*rmse/(2*ext):.1f}% of trajectory extent), median rotation diff {np.median(ang):.1f} deg, scale {s:.3f}")
best = min(res, key=lambda k: res[k][0]); rel, medang, s, R, t, Rc, cV = res[best]
print("best convention:", best)
def cloud(depth, conf, K, c2w, keep=0.5, step=2):
    pts, cols = [], []
    thr = np.quantile(conf[:, ::7, ::7], 1 - keep)
    H, W = depth.shape[1:]
    v, u = np.mgrid[0:H:step, 0:W:step]
    for i in range(len(depth)):
        d = depth[i, ::step, ::step]; m = (conf[i, ::step, ::step] >= thr) & (d > 0)
        z = d[m]; x = (u[m] - K[i, 0, 2]) / K[i, 0, 0] * z; y = (v[m] - K[i, 1, 2]) / K[i, 1, 1] * z
        pts.append((c2w[i, :3, :3] @ np.stack([x, y, z]) + c2w[i, :3, 3:4]).T)
    return np.concatenate(pts)
Sd, Sc = S["depth"].astype(np.float32), S["depth_conf"].astype(np.float32)
Vd, Vc = V["depth"].astype(np.float32), V["depth_conf"].astype(np.float32)
# depth agreement: resample VGGT depth to Solv3R size, apply scale s
dd = []
for i in range(n):
    vd = cv2.resize(Vd[i], (Sd.shape[2], Sd.shape[1]), interpolation=cv2.INTER_LINEAR) * s
    m = (Sc[i] >= np.quantile(Sc[i], 0.5)) & (vd > 0) & (Sd[i] > 0)
    dd.append(np.median(np.abs(vd[m] - Sd[i][m]) / Sd[i][m]))
print(f"median per-pixel relative depth difference (confident pixels, after scale): {np.median(dd)*100:.1f}% (frame-wise median of medians), worst frame {np.max(dd)*100:.1f}%")
# clouds in Solv3R's world
c2w_V = np.zeros((n, 4, 4)); c2w_V[:, 3, 3] = 1
c2w_V[:, :3, :3] = R @ Rc; c2w_V[:, :3, 3] = (s * (R @ cV.T)).T + t       # VGGT cameras expressed in the Solv3R frame
# VGGT depth is in its own scale: multiply by s so it matches
PV = cloud(Vd * s, Vc, V["intrinsics"], c2w_V); PS = cloud(Sd, Sc, S["K"], S["c2w"])
# colours: re-read keyframe images
import glob, os
def render(P, name, axes, lo, hi, size=900):
    a, b = axes; img = np.full((size, size, 3), 255, np.uint8)
    u = ((P[:, a] - lo[0]) / (hi[0] - lo[0]) * (size - 1)).astype(int); w = ((P[:, b] - lo[1]) / (hi[1] - lo[1]) * (size - 1)).astype(int)
    k = (u >= 0) & (u < size) & (w >= 0) & (w < size)
    cnt = np.zeros((size, size), np.int32); np.add.at(cnt, (w[k], u[k]), 1)
    dens = np.clip(np.log1p(cnt) / np.log1p(np.percentile(cnt[cnt > 0], 99)), 0, 1)
    img = (255 - (dens * 255)).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR); cv2.putText(img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
    return img
allp = np.concatenate([PS, PV]); lo_hi = {}
for tag, axes in (("topdown_xz", (0, 2)), ("front_xy", (0, 1))):
    a, b = axes; sel = np.concatenate([PS, PV])[:, [a, b]]
    lo = np.percentile(sel, 1, axis=0); hi = np.percentile(sel, 99, axis=0); m = max(hi - lo); lo = (lo + hi) / 2 - m / 2; hi = lo + m
    cv2.imwrite(f"{out}_{tag}.png", np.hstack([render(PS, "Solv3R", axes, lo, hi), render(PV, "VGGT-Omega", axes, lo, hi)]))
print("points", len(PS), len(PV), "wrote", out + "_*.png")
