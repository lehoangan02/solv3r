"""Windowed offline reconstruction: overlapping windows, each solved offline, chained with similarity transforms.

The frames are cut into windows of --window frames that start every (--window - --overlap) frames. Each window is an
ordinary offline solve (all its frames together, --calls calls), so peak GPU memory is that of a --window-frame offline
solve, whatever the length of the input. Each window has its own world frame (its first image) and its own scale, so
window k is mapped onto the frame of window k-1 by the similarity transform (scale, rotation, translation) that best
aligns the two windows' per-pixel world points of the frames they share (dense correspondences: the same pixel of the
same image in both solves), fitted with trimmed Umeyama. The transforms are chained, so everything ends up in the frame
of the first image, as in offline.py. Each shared frame is taken from the window where it sits more centrally (the
overlap is split at its middle).

Writes <out>/solve.npz (c2w, K, depth, depth_conf, names, window: the window each frame came from), <out>/solve.ply,
<out>/solve.html (simple viewer), <out>/windows.json (frames, transform and alignment residual of every window) and
<out>/note.txt.

usage: python examples/windowed.py --images path/to/images_or_video --out outputs/windowed --window 100 --overlap 50
       python examples/windowed.py --images ... --out ... --vram-gb auto        # window = what fits the free VRAM
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solv3r import Solv3R, list_images, load_images  # noqa: E402
from solv3r import device as D  # noqa: E402
from solv3r.export import scene_points, write_html, write_ply  # noqa: E402
from solv3r.images import model_size  # noqa: E402
from solv3r.model import autocast, cameras_from_pose_enc  # noqa: E402

# peak GPU memory ~ fixed + per_view * views (16 calls), measured on an L40S (same numbers as offline.py)
VRAM_FIXED_GB, VRAM_GB_PER_VIEW = 0.5, 0.0235


def window_starts(n, window, step):
    """Start indices of windows of `window` frames every `step` frames covering n frames; the last window ends at n."""
    if n <= window:
        return [0]
    starts = list(range(0, n - window, step))
    if starts[-1] != n - window:
        starts.append(n - window)
    return starts


def umeyama(src, dst):
    """Similarity (s, R, t) minimising |s R src + t - dst|^2. src, dst (N, 3) float64."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s)
    t = mu_d - s * R @ mu_s
    return s, R, t


def fit_sim3(src, dst, iters=5, trim=0.8):
    """Trimmed Umeyama: refit on the `trim` share of correspondences with the smallest residuals."""
    keep = np.ones(len(src), bool)
    for _ in range(iters):
        s, R, t = umeyama(src[keep], dst[keep])
        res = np.linalg.norm(src @ (s * R).T + t - dst, axis=1)
        keep = res <= np.quantile(res, trim)
    scale = float(np.median(np.linalg.norm(dst - np.median(dst, 0), axis=1)))
    return s, R, t, float(np.median(res[keep])), float(np.median(res[keep]) / max(scale, 1e-9))


def correspondences(p_ref, c_ref, p_new, c_new, conf_q=0.5, max_pts=200_000, seed=0):
    """Pixels confident in both solves: (src from the new window, dst from the reference) points (M, 3)."""
    ok = np.isfinite(p_ref).all(-1) & np.isfinite(p_new).all(-1)
    for c in (c_ref, c_new):
        thr = np.quantile(c.reshape(len(c), -1), conf_q, axis=1)[:, None, None]   # per frame
        ok &= c >= thr
    idx = np.flatnonzero(ok.ravel())
    if len(idx) > max_pts:
        idx = np.random.default_rng(seed).choice(idx, max_pts, replace=False)
    return p_new.reshape(-1, 3)[idx].astype(np.float64), p_ref.reshape(-1, 3)[idx].astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="a folder of images or a video file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--window", type=int, default=None, help="frames per window (default: from --vram-gb, else 100)")
    ap.add_argument("--overlap", type=int, default=None, help="frames shared by consecutive windows (default: window // 2)")
    ap.add_argument("--vram-gb", default=None, help="GPU memory per window in GB, or 'auto' (free memory now); "
                    "window = (vram - 0.5) / 0.0235")
    ap.add_argument("--calls", type=int, default=16)
    ap.add_argument("--adaptive", type=float, default=None)
    ap.add_argument("--max-calls", type=int, default=64)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--keep", type=float, default=1.0, help="fraction of each window's points in the .ply, most confident first")
    ap.add_argument("--sky-mask", action="store_true")
    ap.add_argument("--align-conf", type=float, default=0.5, help="use pixels above this per-frame confidence quantile "
                    "in both windows to fit the transforms")
    a = ap.parse_args()
    t0 = time.time()

    model = Solv3R.from_pretrained(a.checkpoint)
    paths = list_images(a.images, a.stride)
    n = len(paths)
    mem = D.memory_info(model.device)                  # CUDA, Apple MPS (unified memory) or CPU
    free_gb = mem["free_gb"]
    peak = D.PeakMemory(model.device).start()
    budget_gb = None
    W = a.window
    if W is None and a.vram_gb is not None:
        budget_gb = free_gb if a.vram_gb == "auto" else float(a.vram_gb)
        W = max(2, int((budget_gb - VRAM_FIXED_GB) / VRAM_GB_PER_VIEW))
    W = min(W or 100, n)
    O = a.overlap if a.overlap is not None else W // 2
    if not 1 <= O < W and n > W:
        raise SystemExit(f"--overlap must be in [1, window) = [1, {W}), got {O}")
    starts = window_starts(n, W, W - O)
    # each frame belongs to one window: an overlap is split at its middle
    owner_lo = [0] + [(starts[k] + starts[k - 1] + W) // 2 for k in range(1, len(starts))]
    owner_hi = owner_lo[1:] + [n]
    print(f"{n} frames, {len(starts)} windows of {W}, overlap {O}")

    from PIL import Image, ImageOps
    with Image.open(paths[0]) as im:
        size = model_size(*ImageOps.exif_transpose(im).size[::-1])

    names = [os.path.basename(p) for p in paths]
    c2w_all = np.zeros((n, 4, 4), np.float32)
    K_all = np.zeros((n, 3, 3), np.float32)
    depth_all = np.zeros((n, *size), np.float16)
    conf_all = np.zeros((n, *size), np.float16)
    owner = np.full(n, -1, np.int32)
    xyz_parts, rgb_parts, info = [], [], []
    prev = None                                          # previous window: (start, world points in global frame, conf)

    for k, s0 in enumerate(starts):
        tw = time.time()
        idx = list(range(s0, min(s0 + W, n)))
        images = load_images([paths[i] for i in idx], size=size)
        ctx = model.encode(images)
        z = model.start(ctx)
        with torch.no_grad():
            for call in range(1, (a.max_calls if a.adaptive else a.calls) + 1):
                with autocast():
                    z_next = model.step(z, ctx)
                change = float((z_next.float() - z.float()).norm() / z.float().norm())
                z = z_next
                if a.adaptive and call >= 4 and change < a.adaptive:
                    break
            pred = model.decode(z, ctx)
        pts = pred["world_points"].float().cpu().numpy()
        conf = pred["depth_conf"].float().cpu().numpy()
        depth = pred["depth"][..., 0].float().cpu().numpy()
        c2w, K = cameras_from_pose_enc(pred["pose_enc"], tuple(pred["depth"].shape[1:3]))
        c2w, K = c2w.numpy().astype(np.float64), K.numpy()
        del ctx, z, z_next, pred
        D.empty_cache(model.device)

        # similarity transform of this window into the global frame
        if prev is None:
            s, R, t, res, rel = 1.0, np.eye(3), np.zeros(3), 0.0, 0.0
            n_corr = 0
        else:
            p_start, p_pts, p_conf = prev
            shared = range(s0, p_start + len(p_pts))     # frames in both windows
            a_prev = [i - p_start for i in shared]
            a_cur = [i - s0 for i in shared]
            src, dst = correspondences(p_pts[a_prev], p_conf[a_prev], pts[a_cur], conf[a_cur], a.align_conf, seed=k)
            n_corr = len(src)
            s, R, t, res, rel = fit_sim3(src, dst)
        pts = (pts.reshape(-1, 3) @ (s * R).T + t).reshape(pts.shape).astype(np.float32)
        c2w[:, :3, 3] = c2w[:, :3, 3] @ (s * R).T + t
        c2w[:, :3, :3] = R @ c2w[:, :3, :3]
        depth = depth * s

        # this window's own frames
        lo, hi = owner_lo[k] - s0, owner_hi[k] - s0
        g = slice(owner_lo[k], owner_hi[k])
        c2w_all[g], K_all[g] = c2w[lo:hi], K[lo:hi]
        depth_all[g], conf_all[g], owner[g] = depth[lo:hi], conf[lo:hi], k
        xyz, rgb = scene_points(pts[lo:hi], conf[lo:hi], depth[lo:hi], images[lo:hi], a.keep, a.sky_mask)
        xyz_parts.append(xyz)
        rgb_parts.append(rgb)

        prev = (s0, pts, conf)
        info.append(dict(window=k, frames=[s0, s0 + len(idx)], owns=[owner_lo[k], owner_hi[k]], calls=call,
                         last_change=round(change, 5), scale=s, R=R.tolist(), t=t.tolist(), correspondences=n_corr,
                         median_residual=res, median_residual_rel=rel, seconds=round(time.time() - tw, 1)))
        print(f"window {k + 1}/{len(starts)}: frames {s0}-{s0 + len(idx) - 1}, {call} calls, scale {s:.4f}, "
              f"residual {rel * 100:.2f}% of scene radius, {time.time() - tw:.0f} s", flush=True)

    os.makedirs(a.out, exist_ok=True)
    xyz, rgb = np.concatenate(xyz_parts), np.concatenate(rgb_parts)
    np.savez_compressed(f"{a.out}/solve.npz", c2w=c2w_all, K=K_all, depth=depth_all, depth_conf=conf_all,
                        names=np.array(names), window=owner)
    write_ply(f"{a.out}/solve.ply", xyz, rgb)
    write_html(f"{a.out}/solve.html", xyz, rgb, c2w_all, title=os.path.basename(os.path.normpath(a.images)))
    with open(f"{a.out}/windows.json", "w") as f:
        json.dump(dict(window=W, overlap=O, frames=n, windows=info), f, indent=1)
    with open(f"{a.out}/note.txt", "w") as f:
        f.write(f"mode: windowed offline (examples/windowed.py)\n")
        f.write(f"gpu: {D.device_name(model.device)}\n")
        f.write(f"gpu_total_vram_gb: {mem['total_gb']:.1f}\n")
        f.write(f"gpu_free_vram_gb_at_start: {free_gb:.1f}\n")
        f.write(f"vram_budget_gb: {'none' if budget_gb is None else format(budget_gb, '.1f')}\n")
        f.write(f"frames_used: {n}\n")
        f.write(f"window: {W}\noverlap: {O}\nwindows: {len(starts)}\n")
        f.write(f"max_alignment_residual_rel: {max(w['median_residual_rel'] for w in info):.4f}\n")
        f.write(f"points: {len(xyz)}\n")
        f.write(f"peak_torch_vram_gb: {peak.stop():.2f}\n")
        f.write(f"total_seconds: {time.time() - t0:.0f}\n")
        f.write(f"command: {' '.join(sys.argv)}\n")


if __name__ == "__main__":
    main()
