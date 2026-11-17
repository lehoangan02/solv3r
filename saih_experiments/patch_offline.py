p = "/home/lehoangan/LHAN/solv3r_test/solv3r/examples/offline.py"
s = open(p).read()
assert "--num-frames" not in s

s = s.replace("import argparse\nimport os\nimport sys\n", "import argparse\nimport os\nimport sys\nimport time\n\nimport torch\n", 1)

s = s.replace("def main():", "# peak GPU memory ~ fixed + per_view * views (16 calls), measured on an L40S (see the frames-vs-VRAM table)\n"
              "VRAM_FIXED_GB, VRAM_GB_PER_VIEW = 0.5, 0.0235\n\n\ndef main():", 1)

s = s.replace("""    a = ap.parse_args()
""", """    ap.add_argument("--num-frames", type=int, default=None,
                    help="solve exactly this many frames, sampled uniformly from the input (after --stride); overrides --vram-gb")
    ap.add_argument("--vram-gb", default=None,
                    help="GPU memory to use in GB, or 'auto' for the GPU's free memory right now; the number of frames is "
                         "computed from it as (vram - 0.5) / 0.0235")
    a = ap.parse_args()
    t0 = time.time()
""", 1)

s = s.replace("""    paths = list_images(a.images, a.stride)
""", """    paths = list_images(a.images, a.stride)
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
""", 1)

s = s.replace("""    save_solve(f"{a.out}/solve", images, model.decode(z, ctx), names, keep=a.keep, sky=a.sky_mask, view_keep=meta["view_keep"])
""", """    save_solve(f"{a.out}/solve", images, model.decode(z, ctx), names, keep=a.keep, sky=a.sky_mask, view_keep=meta["view_keep"])
    with open(f"{a.out}/note.txt", "w") as f:            # for logging: which GPU / how much VRAM this run had and used
        f.write(f"gpu: {gpu.name}\\n")
        f.write(f"gpu_total_vram_gb: {gpu.total_memory / 1024**3:.1f}\\n")
        f.write(f"gpu_free_vram_gb_at_start: {free_gb:.1f}  (the GPU may be shared with other jobs)\\n")
        f.write(f"vram_budget_gb: {'none' if budget_gb is None else format(budget_gb, '.1f')}\\n")
        f.write(f"frames_available: {total_frames}\\n")
        f.write(f"frames_used: {len(paths)}\\n")
        f.write(f"calls: {call}\\n")
        f.write(f"peak_torch_vram_gb: {torch.cuda.max_memory_allocated() / 1024**3:.2f}\\n")
        f.write(f"total_seconds: {time.time() - t0:.0f}\\n")
        f.write(f"command: {' '.join(sys.argv)}\\n")
""", 1)
open(p, "w").write(s)
print("patched")
