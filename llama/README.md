# Llama inference

Check coverage of the 19 operation fixtures and run the FP32, one-token,
one-layer inference benchmark on Orin. The expert baseline uses ggml CUDA.

```
python3 setup.py ggml
python3 reproduce.py llama --output results/runs/llama
```

Allow 15–40 minutes after [setup](../docs/reproduction.md), excluding the ggml
build. `coverage.json` records recognition and executable lowering separately;
the paper reports 14 of 19 operations. `summary.json` records five timing
samples and compares all 32,000 output logits against ggml with
`atol=0.001, rtol=0.0001`. The paper's LoopLibrarian runtime is 12.826 ms.

The serial CPU output comparison is reported separately and is not expected
to pass the same strict check in every case. The source fixtures in `inputs/`
include the frontend adaptations used by the experiment, including split RoPE
and branchless masking.
