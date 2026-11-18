"""Device and precision policy for CUDA, Apple silicon (MPS) and CPU.

Solv3R was trained and evaluated in bfloat16 autocast on CUDA.  On Apple silicon the GPU runs float16 about twice as
fast as bfloat16 (M1 has no bfloat16 at all), so MPS defaults to float16; the outputs agree with float32 to well within
the model's own error (see the mac port notes).  CPU runs float32.

    dev = best_device()                 # cuda > mps > cpu, or SOLV3R_DEVICE
    set_precision(dev, "auto")          # the autocast used by Solv3R.solve / decode / the streaming solver
    with autocast(): ...
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading
import time

import torch

_DTYPES = {"fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
           "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
           "fp32": torch.float32, "float32": torch.float32, "full": torch.float32}

_POLICY = {"device_type": None, "dtype": None}           # set by set_precision (Solv3R.from_pretrained)


def best_device(prefer: str | None = None) -> torch.device:
    """`prefer` ('auto', 'cuda', 'mps', 'cpu', 'cuda:1', ...) or SOLV3R_DEVICE; auto: cuda > mps > cpu."""
    prefer = prefer or os.environ.get("SOLV3R_DEVICE") or "auto"
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype(device) -> torch.dtype:
    t = torch.device(device).type
    if t == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if t == "mps":
        return torch.float16
    return torch.float32


def parse_dtype(precision, device) -> torch.dtype:
    """'auto' | 'fp16' | 'bf16' | 'fp32' (or a torch.dtype) -> the compute dtype on `device`."""
    if isinstance(precision, torch.dtype):
        return precision
    if precision in (None, "auto"):
        return default_dtype(device)
    try:
        return _DTYPES[str(precision).lower()]
    except KeyError:
        raise ValueError(f"unknown precision {precision!r}: use auto, fp16, bf16 or fp32") from None


def set_precision(device, precision="auto"):
    """Sets the autocast policy used by autocast(); returns the compute dtype."""
    device = torch.device(device)
    dtype = parse_dtype(precision, device)
    _POLICY["device_type"], _POLICY["dtype"] = device.type, dtype
    return dtype


def policy():
    if _POLICY["device_type"] is None:                    # nobody chose: the best device, its default precision
        set_precision(best_device())
    return _POLICY["device_type"], _POLICY["dtype"]


def autocast():
    """Autocast in the chosen precision (a no-op in float32)."""
    device_type, dtype = policy()
    if dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device_type, dtype=dtype, cache_enabled=False)


def no_autocast(x=None):
    """Autocast off (float32 regions such as the output layers), on the device of tensor x (or the policy device)."""
    device_type = x.device.type if isinstance(x, torch.Tensor) else policy()[0]
    if device_type not in ("cuda", "cpu", "mps", "xpu"):
        return contextlib.nullcontext()
    return torch.autocast(device_type, enabled=False)


# ---------------------------------------------------------------------- memory and timing
def sync(device=None):
    t = torch.device(device).type if device is not None else policy()[0]
    if t == "cuda":
        torch.cuda.synchronize()
    elif t == "mps":
        torch.mps.synchronize()


def empty_cache(device=None):
    t = torch.device(device).type if device is not None else policy()[0]
    if t == "cuda":
        torch.cuda.empty_cache()
    elif t == "mps":
        torch.mps.empty_cache()


def allocated_gb(device=None) -> float:
    t = torch.device(device).type if device is not None else policy()[0]
    if t == "cuda":
        return torch.cuda.memory_allocated() / 2**30
    if t == "mps":
        return torch.mps.current_allocated_memory() / 2**30
    return 0.0


def device_name(device=None) -> str:
    device = torch.device(device) if device is not None else torch.device(policy()[0])
    if device.type == "cuda":
        return torch.cuda.get_device_name(device).replace("NVIDIA ", "")
    if device.type == "mps":
        import platform
        import subprocess
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        except OSError:
            chip = platform.processor()
        return f"{chip or 'Apple silicon'} GPU (MPS)"
    return "CPU"


def memory_info(device=None) -> dict:
    """total_gb: what the device can give PyTorch (CUDA: card memory; MPS: Metal's recommended working set, a share of
    the unified memory); free_gb: free now (CUDA) or total minus what this process holds (MPS)."""
    device = torch.device(device) if device is not None else torch.device(policy()[0])
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        return dict(total_gb=total / 2**30, free_gb=free / 2**30)
    if device.type == "mps":
        total = torch.mps.recommended_max_memory() / 2**30
        return dict(total_gb=total, free_gb=max(0.0, total - torch.mps.driver_allocated_memory() / 2**30))
    try:
        import os as _os
        total = _os.sysconf("SC_PAGE_SIZE") * _os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        total = 0.0
    return dict(total_gb=total, free_gb=total)


def available_ram_gb():
    """RAM other programs leave free right now (macOS: free + inactive + purgeable + speculative pages from vm_stat;
    Linux: MemAvailable).  On a Mac the GPU uses this same memory: a run that needs more makes macOS swap."""
    import subprocess
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page = int(out.split("page size of ")[1].split(" ")[0])
            pages = {l.split(":")[0].strip(): int(l.split(":")[1].strip().rstrip(".")) for l in out.splitlines()[1:] if ":" in l}
            n = sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages purgeable", "Pages speculative"))
            return n * page / 2**30
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except Exception:  # noqa: BLE001
        pass
    return float("nan")


def set_memory_cap(gb, device=None):
    """Hard cap on PyTorch's allocator: allocations past `gb` raise an out-of-memory error instead of swapping (MPS) or
    taking a shared GPU (CUDA).  Returns the fraction set, or None on CPU."""
    device = torch.device(device) if device is not None else torch.device(policy()[0])
    if gb is None:
        return None
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory / 2**30
        frac = min(1.0, gb / total)
        torch.cuda.set_per_process_memory_fraction(frac, device)
        return frac
    if device.type == "mps":
        total = torch.mps.recommended_max_memory() / 2**30
        frac = min(2.0, gb / total)                       # MPS allows up to 2x the recommended working set
        torch.mps.set_per_process_memory_fraction(frac)
        return frac
    return None


class PeakMemory:
    """Peak memory of a block.  CUDA keeps the true peak; MPS has no peak counter, so a thread samples the Metal driver's
    allocation every `interval` seconds (the peak of short spikes can be missed by a little).

        with PeakMemory(dev) as pm: ...
        pm.peak_gb
    """

    def __init__(self, device=None, interval=0.05):
        self.device = torch.device(device) if device is not None else torch.device(policy()[0])
        self.interval = interval
        self.peak_gb = 0.0                                # peak of the tensors
        self.held_gb = 0.0                                # peak of what the allocator held (tensors + its cache)
        self._stop = threading.Event()
        self._thread = None

    def sample(self):
        if self.device.type == "cuda":
            self.peak_gb = max(self.peak_gb, torch.cuda.max_memory_allocated() / 2**30)
            self.held_gb = max(self.held_gb, torch.cuda.max_memory_reserved() / 2**30)
        elif self.device.type == "mps":                  # tensors, and all the Metal driver holds (tensors + cache)
            self.peak_gb = max(self.peak_gb, torch.mps.current_allocated_memory() / 2**30)
            self.held_gb = max(self.held_gb, torch.mps.driver_allocated_memory() / 2**30)
        return self.peak_gb

    def _run(self):
        while not self._stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        elif self.device.type == "mps":
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        self.sample()
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
        self.sample()
        return False

    start = __enter__

    def stop(self):
        self.__exit__(None, None, None)
        return self.peak_gb


class Timer:
    """Wall-clock seconds of a block, synchronized with the device."""

    def __init__(self, device=None):
        self.device = device
        self.seconds = 0.0

    def __enter__(self):
        sync(self.device)
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync(self.device)
        self.seconds = time.perf_counter() - self._t
        return False
