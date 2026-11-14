# Repository layout

- `polybench/`, `llama/`, `aten/`, `mfem/`, and `pva/`: benchmark sources,
  harnesses, manifests, and reproduction drivers.
- `eqsat/`, `graph_sizes/`, `ginsbach/`, and `kernelfarer/`: comparison experiments.
- `compilation_cost/` and `library_inventory/`: compilation measurements and
  library specification counts.
- `inputs/`: additional extracted sources and shared benchmark utilities.
- `scripts/artifact/`: shared configuration, command recording, and SSH/local
  execution helpers.
- `scripts/correctness/`: source preparation, compilation, and output checks
  used by the experiment drivers.
- `third_party/`: the pinned Polygeist submodule and installed dependencies.
- `dependency_sources/`: revisions and setup files for public dependencies.
- `tests/`: checks for the reproduction scripts and benchmark manifests.
- `results/runs/`: generated results, ignored by Git.

The top-level `reproduce.py` selects an experiment. Each experiment's README
explains its inputs, command, and outputs. Compiler code, library references,
and API bindings are in `third_party/Polygeist`.
