import sys, runpy, atexit, torch
torch.cuda.set_per_process_memory_fraction(0.33)   # ~14.3 GiB of the 46 GB L40S
atexit.register(lambda: print("peak allocated GiB:", torch.cuda.max_memory_allocated()/2**30, "reserved:", torch.cuda.max_memory_reserved()/2**30, flush=True))
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
