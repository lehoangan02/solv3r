# Solv3R offline mode: how many frames fit in how much GPU memory

Measured on an NVIDIA L40S (44.4 GiB usable, 46068 MiB reported by nvidia-smi), Solv3R from `ubc-vision/solv3r` at commit
`c3851d3` (Nov 2026), `examples/offline.py`-equivalent pipeline: `encode` -> 16 solver calls -> `decode`,
default settings, released weights, images 504 px on the long side (16:9 frames from `I87X/ScanVideos/Video1.MOV`,
sampled uniformly).

"Fits" means the full pipeline completed with `torch.cuda.set_per_process_memory_fraction` set to the budget (a hard
cap on PyTorch's allocator), so a trial that needs more memory raises `torch.cuda.OutOfMemoryError`.

## Frames that fit

| GPU memory budget | Largest N that fit | Smallest N that failed | Peak allocated at the largest fit | Solve time at the largest fit |
|---|---|---|---|---|
| 4 GB  | 152  | 156  | 3.60 GB  | 8 s |
| 8 GB  | 320  | 336  | 7.09 GB  | 24 s |
| 16 GB | 660  | 690  | 14.14 GB | 112 s |
| 32 GB | 1340 | 1440 | 28.25 GB | 467 s |
| Full L40S (about 44 GiB) | 1856 | 1920 | 38.96 GB | 924 s |

The true limit lies between the two columns; the search stopped at about 5% resolution (the 32 GB row was only
bracketed at 1340 / 1440 / 1500, not bisected further). Times are wall-clock for model load + solve on an otherwise idle
GPU. Time grows faster than linearly in N.

## Rule of thumb

- Peak allocated memory is almost linear: about `0.5 GB + 0.0207 GB x N` (N=128: 3.1 GB, 512: 11.1 GB, 1024: 21.7 GB, 1792: 37.6 GB).
- A budget needs about 10% headroom above the peak (allocator overhead), so the number of frames that fit is
  about `(budget_GB - 0.5) / 0.0235`. `examples/offline.py --vram-gb G` uses exactly this formula.
  It matches the table (4 GB -> 148, 8 GB -> 319, 16 GB -> 659, 32 GB -> 1340; the full card, 43.5 GB free -> 1829).
- Without `--vram-gb`/`--num-frames` all frames are loaded, as before.

## New arguments in `examples/offline.py`

```
--num-frames N     solve exactly N frames, sampled uniformly from the input (after --stride); overrides --vram-gb
--vram-gb G|auto   GPU memory to use in GB, or 'auto' = the GPU's free memory right now; N = int((G - 0.5) / 0.0235)
```

Every run also writes `note.txt` in the output folder: GPU name, total VRAM, free VRAM at start, the VRAM budget used,
frames available/used, calls, peak PyTorch VRAM, total seconds and the command line.

## The runs on the RoSE scans (`--vram-gb auto`, GPU idle, 43.5 GB free)

| Room (video) | Frames in video | Frames used | Peak PyTorch VRAM | Total time |
|---|---|---|---|---|
| Bedroom1 `Video1.mp4` | 742 | 742 (all) | 15.84 GB | 199 s |
| P64 `Video1.MOV` | 1918 | 1829 | 38.40 GB | 1027 s |
| I87X `Video1.MOV` | 7910 | 1829 (uniform) | 38.40 GB | 1197 s |

Bedroom1 uses all of its frames, so it is limited by the video, not the GPU. P64 and I87X are limited by VRAM.
Outputs are in `<Room>/Solv3r2_ScanVideo1/` (`solve.ply`, `solve.npz`, `solve.html`, `note.txt`, `run.log`,
`monitor.log`, `frames/`). Note that `monitor.log` (nvidia-smi) reads higher than the real need because PyTorch's allocator
caches memory; `peak_torch_vram_gb` in `note.txt` is the number to use.

## Caveats

- The `auto` count is conservative by design (about 1.5% below the measured limit of 1856 for the full card). The
  full-card probes were not run in isolation from other GPU activity in every case, so treat 1856 as approximate
  (it failed at 1920).
- The table was measured at 16 calls. Fewer calls do not reduce peak memory much (it is dominated by the number of views);
  other resolutions (not 504 px on the long side) change the per-view cost.
- Other jobs on the GPU reduce what is free; `--vram-gb auto` reads the free memory at start.
