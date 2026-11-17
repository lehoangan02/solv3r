"""Solv3R (617 kf) vs VGGT-Omega windowed (617 kf) vs VGGT-Omega single pass (100 kf): all exported with the same scene_points filter, aligned by camera trajectory, rendered from several views."""
import sys, os, glob, hashlib, numpy as np, cv2
out, all_dir = sys.argv[1], sys.argv[2]
specs = [a.split(",") for a in sys.argv[3:]]            # label,npz,ply,imgdir,kind(c2w|w2c)
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
allf = sorted(glob.glob(all_dir + "/*.jpg")); idx_of = {md5(p): i for i, p in enumerate(allf)}
def load(label, npz, ply, imdir, kind):
    Z = np.load(npz); names = Z["names"]; oi = np.array([idx_of[md5(os.path.join(imdir, str(n)))] for n in names])
    if kind == "w2c": E = Z["extrinsics"].astype(np.float64); Rc = E[:, :, :3].transpose(0, 2, 1); cen = -(Rc @ E[:, :, 3][..., None])[..., 0]
    else: Rc = Z["c2w"][:, :3, :3].astype(np.float64); cen = Z["c2w"][:, :3, 3].astype(np.float64)
    o = np.argsort(oi); return dict(label=label, oi=oi[o], cen=cen[o], Rc=Rc[o], ply=ply, c2w_raw=Z["c2w"] if kind == "c2w" else None)
M = [load(*s) for s in specs]
def read_ply(p, cap=6_000_000):
    with open(p, "rb") as f:
        while True:
            l = f.readline()
            if l.startswith(b"element vertex"): n = int(l.split()[2])
            if l.startswith(b"end_header"): break
        a = np.frombuffer(f.read(), dtype=[("p", "<f4", 3), ("c", "u1", 3)], count=n)
    tot = n
    if n > cap: a = a[np.random.default_rng(0).choice(n, cap, replace=False)]
    return a["p"].copy(), a["c"].copy(), tot
def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0); X, Y = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(Y.T @ X / len(src)); Sg = np.eye(3); Sg[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ Sg @ Vt; s = (D * Sg.diagonal()).sum() / X.var(0).sum(); return s, R, mu_d - s * R @ mu_s
ref = M[0]; clouds = []
for m in M:
    P, C, tot = read_ply(m["ply"]); m["tot"] = tot
    if m is ref: clouds.append((P, C)); m["s"], m["R"], m["t"] = 1.0, np.eye(3), np.zeros(3); continue
    cr = np.stack([np.interp(m["oi"], ref["oi"], ref["cen"][:, k]) for k in range(3)], 1)
    s, R, t = umeyama(m["cen"], cr); al = (s * (R @ m["cen"].T)).T + t
    ext = np.linalg.norm(cr - cr.mean(0), axis=1).max(); rmse = np.sqrt(((al - cr) ** 2).sum(1).mean())
    print(f'{m["label"]}: camera-centre RMSE vs {ref["label"]} {100*rmse/(2*ext):.2f}% of extent, scale {s:.3f}')
    clouds.append(((s * (R @ P.T)).T + t, C)); m["s"], m["R"], m["t"] = s, R, t
for m, (P, C) in zip(M, clouds): print(f'{m["label"]}: {m["tot"]} points in ply (rendering {len(P)})')
allp = np.concatenate([P for P, _ in clouds]); lo, hi = np.percentile(allp, 1, axis=0), np.percentile(allp, 99, axis=0); ext_scene = (hi - lo).max(); cen = (lo + hi) / 2
vox = ext_scene * 0.015
def occ(P):
    m = (P >= lo - 0.05 * ext_scene).all(1) & (P <= hi + 0.05 * ext_scene).all(1)
    return set(map(tuple, np.unique(np.floor(P[m] / vox).astype(np.int32), axis=0)))
occs = [occ(P) for P, _ in clouds]
for m, o in zip(M[1:], occs[1:]): print(f'cells: {ref["label"]} {len(occs[0])}, {m["label"]} {len(o)}; {ref["label"]} cells also in it {100*len(occs[0]&o)/len(occs[0]):.0f}%, its cells also in {ref["label"]} {100*len(occs[0]&o)/len(o):.0f}%')
SZ = 600
def ortho(P, C, a, b, flip):
    u = ((P[:, a] - (cen[a] - ext_scene / 2)) / ext_scene * (SZ - 1)).astype(int); w = ((P[:, b] - (cen[b] - ext_scene / 2)) / ext_scene * (SZ - 1)).astype(int)
    if flip: w = SZ - 1 - w
    k = (u >= 0) & (u < SZ) & (w >= 0) & (w < SZ); o = np.argsort(-P[k][:, 3 - a - b])
    img = np.full((SZ, SZ, 3), 255, np.uint8); img[w[k][o], u[k][o]] = C[k][o][:, ::-1]; return img
def persp(P, C, c2w, f=0.9):
    w2c = np.linalg.inv(c2w); X = P @ w2c[:3, :3].T + w2c[:3, 3]; k = X[:, 2] > 0.05 * ext_scene; X, Ck = X[k], C[k]
    u = (f * SZ * X[:, 0] / X[:, 2] + SZ / 2).astype(int); w = (f * SZ * X[:, 1] / X[:, 2] + SZ / 2).astype(int)
    kk = (u >= 0) & (u < SZ) & (w >= 0) & (w < SZ); o = np.argsort(-X[kk][:, 2])
    img = np.full((SZ, SZ, 3), 255, np.uint8); img[w[kk][o], u[kk][o]] = Ck[kk][o][:, ::-1]; return img
c2w_ref = ref["c2w_raw"]; n = len(c2w_ref); order = np.argsort(np.argsort(0 * np.arange(n)))   # ref cameras in file order
ids = [int(n * q) for q in (0.25, 0.75)]
def vc(i, back=0.5): c = c2w_ref[i].astype(np.float64).copy(); c[:3, 3] -= c[:3, 2] * back * ext_scene; return c
rows = []
for m, (P, C) in zip(M, clouds):
    ims = [ortho(P, C, 0, 2, True), ortho(P, C, 0, 1, False), ortho(P, C, 2, 1, False)] + [persp(P, C, vc(i)) for i in ids]
    for im, t_ in zip(ims, ["top-down", "front", "side", "persp 25%", "persp 75%"]): cv2.putText(im, t_, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
    cv2.putText(ims[0], m["label"], (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 0, 0), 2); rows.append(np.hstack(ims))
cv2.imwrite(out, np.vstack(rows)); print("wrote", out)
