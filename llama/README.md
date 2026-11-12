# Llama

From the repository root:

```
python3 reproduce.py llama --stage plan
python3 reproduce.py llama --output results/runs/llama
```

Use a fresh output directory for each build. Consult `docs/reproduction.md` for
the exact paper cohort, machine, timing boundary, and correctness gate. Saved
results under `results/archive/` are comparison evidence, not replacement inputs
for compilation or execution. `--cases` selects an explicit subset.
