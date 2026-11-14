# KernelFaRer comparison — Table 9

Run complete FP32 LARGE programs for GEMM, SYR2K, 2mm, and 3mm on the x86 host.
Compare native compilation, KernelFaRer, and LoopLibrarian.

```
python3 setup.py kernelfarer
python3 reproduce.py kernelfarer --output results/runs/table9
```

Allow 30–90 minutes after [setup](../docs/reproduction.md). All four programs
must match the native output with `atol=0.011, rtol=0.0005`, following the
paper's two-decimal output convention. Each implementation uses five warmups
and five timing samples; the reported time is the minimum.

`summary.csv` contains the comparison and `timing.json` retains
raw samples. In GEMM/SYR2K/2mm/3mm order, the paper reports LoopLibrarian speedups
of 8.6/366/115/93 and KernelFaRer speedups of 8.6/346/113/112. The paper's native
baseline used one warmup and ten separate process runs, so its timing protocol
differs from this driver's common five-sample protocol. Host hardware and load
also affect speedups.
