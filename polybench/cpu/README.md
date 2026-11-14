# PolyBench CPU — Figure 3

Run the 17 FP64 LARGE programs on the Orin CPU, pinned to core 0 with one
OpenBLAS thread. Compare the original C programs with the complete programs
compiled by LoopLibrarian and linked against OpenBLAS.

```
python3 reproduce.py polybench_cpu --output results/runs/fig3
```

Allow 40 minutes–2 hours after setup. Each program uses five warmups and five
measurements; the reported time is the minimum. All output elements must pass
`atol=rtol=1e-8`. `results.json` contains the checks, raw samples, and paper
speedups; `RESULTS.csv` contains the timing table.

Select a program with `--cases gemm`. For a small compilation/execution check,
add `--target host --size MINI`; these settings do not reproduce Figure 3 timing.
See the [setup guide](../../docs/reproduction.md) for dependencies and configuration.
