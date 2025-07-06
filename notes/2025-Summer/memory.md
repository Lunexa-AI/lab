Observations and Key Lessons

1. NumPy curve flat-lined because np.empty reserves address space but doesn’t touch pages, so the OS never commits physical RAM.
 - Memory profilers ( %mprun, memory_profiler ) track resident pages only. If you don’t write to the array, it looks “free.” Always touch pages (arr.fill(0) / zero_()) when you want true footprint.

2. PyTorch curve rose and stayed high: its caching allocator grabs pages and keeps them for reuse, even after tensors go out of scope. NumPy gives pages back to malloc/OS immediately (unless you keep the array alive)
- Training loops that repeatedly allocate tensors may appear to “leak” in RSS, really it’s the caching allocator. Don’t chase phantom leaks; instead monitor allocator stats (torch.cuda.memory_summary()).

3. When memory_usage(..., retval=False) returned, Python’s GC dropped the list , NumPy memory vanished.
- Keep objects alive (or set retval=True) while sampling; otherwise your profiler sees deallocation, not peak usage.

4. Torch CPU tensors showed up in RAM; had they been .to("cuda"), RSS would barely change because VRAM isn’t counted.
- Always combine host RAM tools with nvidia-smi, torch.cuda.memory_allocated() or tracemalloc equivalents when profiling GPU workloads.

5. I sampled every 0.1 s; very short-lived peaks might disappear
- For GPU kernels or micro-allocations, tighten interval or use dedicated profilers (PyTorch profiler, cuda-memcheck).

