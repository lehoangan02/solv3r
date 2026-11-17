"""Offline reconstruction: all images are solved together.

Writes <out>/solve.npz (camera-to-world in the frame of the first image, intrinsics, depth, confidence at the model
resolution), <out>/solve.ply and <out>/solve.html (the solve in a browser viewer, played call by call). With --adaptive,
calls stop once a call changes the state by less than that fraction.

usage: python examples/offline.py --images path/to/images --out outputs/offline [--calls 16 | --adaptive 0.02]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solv3r import Solv3R, list_images, load_images  # noqa: E402
from solv3r.export import save_solve  # noqa: E402
from solv3r.model import autocast  # noqa: E402
from solv3r.viewer import Recorder, save_solve_viewer  # noqa: E402


# peak GPU memory ~ fixed + per_view * views (16 calls), measured on an L40S (see the frames-vs-VRAM table)
VRAM_FIXED_GB, VRAM_GB_PER_VIEW = 0.5, 0.0235


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="a folder of images or a video file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None, help="a .safetensors file or a Hugging Face repo id (default: checkpoints/ if present, else the released weights)")
    ap.add_argument("--calls", type=int, default=16)
    ap.add_argument("--adaptive", type=float, default=None)
    ap.add_argument("--max-calls", type=int, default=64)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--keep", type=float, default=1.0, help="fraction of the points in the .ply, the most confident first "
                    "(1: all; points on depth edges always leave), as the demo's confidence filtering")
    ap.add_argument("--sky-mask", action="store_true", help="leave the sky out of the point cloud (outdoor captures)")
    ap.add_argument("--num-frames", type=int, default=None,
                    help="solve exactly this many frames, sampled uniformly from the input (after --stride); overrides --vram-gb")
    ap.add_argument("--vram-gb", default=None,
                    help="GPU memory to use in GB, or 'auto' for the GPU's free memory right now; the number of frames is "
                         "computed from it as (vram - 0.5) / 0.0235")
    a = ap.parse_args()
    t0 = time.time()

    model = Solv3R.from_pretrained(a.checkpoint)
    paths = list_images(a.images, a.stride)
    total_frames = len(paths)
    gpu = torch.cuda.get_device_properties(0)
    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    budget_gb = None
    n = a.num_frames
    if n is None and a.vram_gb is not None:
        budget_gb = free_gb if a.vram_gb == "auto" else float(a.vram_gb)
        n = max(2, int((budget_gb - VRAM_FIXED_GB) / VRAM_GB_PER_VIEW))
    if n is not None and n < len(paths):
        paths = [paths[int(i * len(paths) / n)] for i in range(n)]
    images = load_images(paths)
    ctx = model.encode(images)
    z = model.start(ctx)
    rec = Recorder(images)                               # snapshots of the solve for the viewer
    rec.cameras(model, z, ctx, 0)
    for call in range(1, (a.max_calls if a.adaptive else a.calls) + 1):
        with autocast():
            z_next = model.step(z, ctx)
        change = float((z_next.float() - z.float()).norm() / z.float().norm())
        z = z_next
        rec.record(model, z, ctx, call)
        if a.adaptive and call >= 4 and change < a.adaptive:
            break
    rec.close(model, z, ctx, call)
    print(f"{len(paths)} images, {call} calls, last relative change {change:.4f}")
    os.makedirs(a.out, exist_ok=True)
    names = [os.path.basename(p) for p in paths]
    meta = save_solve_viewer(f"{a.out}/solve.html", rec, names, sky=a.sky_mask, title=os.path.basename(os.path.normpath(a.images)),
                             stats=[f"{len(paths)} views", f"{call} calls"])
    save_solve(f"{a.out}/solve", images, model.decode(z, ctx), names, keep=a.keep, sky=a.sky_mask, view_keep=meta["view_keep"])
    with open(f"{a.out}/note.txt", "w") as f:            # for logging: which GPU / how much VRAM this run had and used
        f.write(f"gpu: {gpu.name}\n")
        f.write(f"gpu_total_vram_gb: {gpu.total_memory / 1024**3:.1f}\n")
        f.write(f"gpu_free_vram_gb_at_start: {free_gb:.1f}  (the GPU may be shared with other jobs)\n")
        f.write(f"vram_budget_gb: {'none' if budget_gb is None else format(budget_gb, '.1f')}\n")
        f.write(f"frames_available: {total_frames}\n")
        f.write(f"frames_used: {len(paths)}\n")
        f.write(f"calls: {call}\n")
        f.write(f"peak_torch_vram_gb: {torch.cuda.max_memory_allocated() / 1024**3:.2f}\n")
        f.write(f"total_seconds: {time.time() - t0:.0f}\n")
        f.write(f"command: {' '.join(sys.argv)}\n")


if __name__ == "__main__":
    main()
