# Compilation cost — Table 8

Measure C-to-linked-AArch64 compilation sequentially, with five repetitions per
input both with and without equality-saturation rewrites.

```
python3 reproduce.py compilation_cost --output results/runs/compilation-cost
```

`inputs.json` lists 199 inputs: 30 PolyBench, 23 Llama, 115 ATen, and 31 MFEM.
The 23 Llama inputs comprise 19 operation fixtures, extended forward, and three
public llama2.c functions. ATen uses the paper dimensions in both configurations.

The full run comprises 1,990 builds. Allow approximately 30–60 hours after
[setup](../docs/reproduction.md), based on initial 50–170-second build probes.
A small check is:

```
python3 reproduce.py compilation_cost --cases aten/aten_avg_pool2d --repetitions 1 --output results/runs/compilation-check
```

`compilation-table.csv` summarizes per-input median build times by suite.
`measurement-tables.csv` reports stage times, proof measurements, and peak
process memory; `summary.json` includes raw repetitions and failures. Inputs
enter a table only when all required repetitions link successfully; excluded
counts are reported. Proofs have a ten-second deadline, and timed-out proofs
cannot authorize replacements.

The paper used three repetitions. `first-three-compilation-table.csv` reports
repetitions 1–3 separately. These measurements include runtime-object generation
and linking; agreement with the paper's measurement boundary and input set
remains unverified.

Use the suite driver with `--resume` to continue a stopped run with identical
inputs and compiler settings, or `--stage summarize` to report available data.
Do not change the compiler during a run. Application compilation is repeated;
a parsed reference-catalog cache is reused separately for each configuration.
Peak RSS is the maximum reported for an individual process, not the sum of
simultaneously resident worker processes.
