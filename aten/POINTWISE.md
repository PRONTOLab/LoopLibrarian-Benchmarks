# Pointwise experiments

`pointwise_candidates.csv` lists 268 candidates from the 598 C fixtures, including
54 of the 115 Figure 5 cases. It records source hashes and operation categories;
being listed does not imply a successful match or executable GPU replacement.

```
python3 aten/pointwise.py --cases exp2 expm1 erfc --output results/runs/pointwise
```

Omit `--cases` to process the full inventory. Use `--stage match`, `build`, `run`,
or `report` to run individual stages. Start matching in a new output directory;
later stages reuse that directory. Configure the compiler, CUDA, LibTorch, and
SSH/local execution as described in the [ATen guide](README.md). Jiterator
operations also require CUDA 12.6 NVRTC and matching `libnvrtc-builtins` libraries
on the target, included in `remote_library_path`.

Allow 2–10 minutes for a selected case, or 20–60 minutes for matching the full
inventory, plus compilation and GPU time for executable cases. `summary.json`
separates candidates, complete matches, strict GPU passes, and timings.

A complete match requires imported-reference proof, valid API arguments, and no
remaining computation loops. Matching defaults to `--numeric-model strict`.
The optional `relaxed` model permits finite-arithmetic transformations such as
reassociation and disregard of signed zero; it does not establish full IEEE
floating-point equivalence. Reports record this choice separately from runtime
correctness. Use separate output directories for these models.

Execution compares every output with both the C fixture and actual ATen CUDA,
checks unchanged inputs and buffer guards, and tests exceptional values and
signed zeros. Five synchronized timing pairs are collected only after strict
checks pass. An unavailable native API or failed comparison is reported. For
example, C `roundf` rounds ties away from zero, whereas ATen round uses ties to
even, so these operations are not interchangeable.

These runs use the original fixture dimensions, which differ from Figure 5.
For the Figure 5 experiment, use the [paper command](README.md). Optional larger
dimensions for 166 candidates are recorded in `pointwise_shapes.json`:

```
python3 aten/pointwise.py --shape-specs aten/pointwise_shapes.json --cases exp2 expm1 erfc --output results/runs/pointwise-large
```

To combine disjoint runs, use `--stage report --extra-gpu-results PATH` with
another run's `gpu/gpu-results.json`. The report checks compatible inputs and
compiler settings and rejects duplicate cases. Results at different dimensions
remain separate.
