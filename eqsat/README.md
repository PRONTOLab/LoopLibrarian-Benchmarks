# Equality saturation comparison

Compile all 30 PolyBench programs with equality saturation enabled and disabled,
then compare recognition with SpEQ.

```
python3 setup.py speq
python3 reproduce.py eqsat --output results/runs/eqsat
```

Allow 30 minutes–2 hours after [setup](../docs/reproduction.md), excluding the
SpEQ download and build. Docker is required; a GPU is not.

`summary.json` reports recognition and executable lowering separately. For the
paper's benchmark set, expect 17 recognized programs with equality saturation
and 8 without. Additional recognitions are listed separately. SpEQ's reported
single match uses GEMM with linearized indexing; results on the unmodified
PolyBench sources are also recorded.
