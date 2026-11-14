# MFEM execution

Run the 18 supported contraction fixtures with `NE=1024` and the integrated
PCG updates with `NE=2` on Orin.

```
python3 reproduce.py mfem --output results/runs/mfem
```

Allow 15–45 minutes after [setup](../docs/reproduction.md). The driver compiles
the C fixtures in `inputs/`, emits GPU library calls, and checks every output
against the original computation with tolerance `1e-11 + 1e-10 * abs(expected)`.

`summary.json` contains aggregate results. Each case has `result.json` and
`run.log` containing its output checks. This driver validates execution; its
cuBLAS call count is not the paper's count of 128 cuTensorNet contraction sites.
No MFEM timing comparison is required by this experiment.
