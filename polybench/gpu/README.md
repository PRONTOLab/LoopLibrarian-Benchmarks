# PolyBench GPU — Figure 4

Run the 12 FP64 LARGE kernels on Orin, comparing native CUDA with
LoopLibrarian-generated library calls.

```
python3 reproduce.py polybench_gpu --output results/runs/fig4
```

Allow 20–50 minutes after [setup](../../docs/reproduction.md). Each case uses
five warmups and five measured samples, with the minimum CUDA-event compute
time reported. This interval excludes allocation and initial/final transfers;
synchronized end-to-end time is recorded separately. All outputs must pass
`atol=rtol=1e-8` before timings are accepted.

`results.csv` contains timings and speedups. Per-case files retain output
checks and a separate vendor-call trace from the timed executable. These
experiments use separate kernel harnesses. Ten native computation bodies come
from pinned PolyBenchGPU; correlation and covariance use the paper's local
CUDA adaptations. Select a subset with `--cases gemm`.
