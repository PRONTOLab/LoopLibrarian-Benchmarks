# Mfem

Sources and harnesses are ordinary files in `mfem/inputs/`.
`mfem/inputs/manifest.json` selects the 18 supported fixtures; the driver also
includes the integrated PCG application. The full-output harness is under
`mfem/inputs/benchmarks/`.

From the repository root:

```
python3 reproduce.py mfem --stage plan
python3 reproduce.py mfem --output results/runs/mfem
```

Use a fresh output directory for each build. Consult `docs/reproduction.md` for
the exact paper cohort, machine, timing boundary, and correctness gate. Saved
results under `results/archive/` are comparison evidence, not replacement inputs
for compilation or execution. `--cases` selects an explicit subset.
