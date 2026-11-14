# Extending the experiments

Complete the [setup](reproduction.md) first. Run commands from the repository
root and use a new output directory for each experiment.

## Select cases or change dimensions

Run one PolyBench case with a smaller dataset on the host:

```
python3 reproduce.py polybench_cpu --cases gemm --target host --size MINI --output results/runs/gemm-mini
```

Supported sizes are `MINI`, `SMALL`, `MEDIUM`, and `LARGE`. The paper uses
FP64 `LARGE` inputs on Orin. Select an ATen case with:

```
python3 reproduce.py aten --cases aten_mm --output results/runs/aten-mm
```

ATen uses dimensions and dtypes from [paper_shapes.json](../aten/paper_shapes.json).
To change dimensions, copy this manifest and update the case's `dims` and
related aliases consistently. Pass the copy through `--shape-specs` to both
[aten_paper115_campaign.py](../scripts/correctness/aten_paper115_campaign.py)
and [aten_fixture_batch.py](../scripts/correctness/aten_fixture_batch.py).
[aten/reproduce.py](../aten/reproduce.py) shows their invocation and environment.
The top-level ATen command does not expose this option. Label modified inputs
separately when comparing results with the paper.

## Add a benchmark

1. Add its C source, origin, license, dimensions, dtype, and any frontend
   adaptations to the relevant experiment directory.
2. Add a harness that initializes valid inputs and compares every output with
   the original program. For performance comparisons, call the real baseline
   implementation and use the same timing boundary for both programs.
3. Register the case in the driver. For ATen, provide a CSV following
   [all_cases.csv](../aten/all_cases.csv), shape and argument metadata, and a
   native ATen invocation. Both lower-level drivers accept `--checklist`.
4. Run compilation and output checking before collecting timings. Inspect the
   generated library calls as well as the match report.

## Add a library reference

References and runtime bindings belong to the Polygeist submodule:

1. Add the reference under `generic_solver/lib_v3/ref_c/` and its manifest
   under `generic_solver/lib_v3/manifest/`. Record the public source, argument
   types and layouts, numerical assumptions, and target API. See the
   [Thrust fill reference](../third_party/Polygeist/generic_solver/lib_v3/ref_c/thrust_fill.c)
   and [manifest](../third_party/Polygeist/generic_solver/lib_v3/manifest/thrust_fill.json).
2. Add a runtime adapter if the target API is not already supported. It must
   call the library with the declared argument bindings and linkage.
3. Run `python3 setup.py specs` to raise and validate the reference. The same
   raising and equality-saturation pipeline compares callers with this IR.
4. Add execution tests for the supported shapes, layouts, and numerical
   assumptions. Unsupported inputs should be rejected.

Preparing the reference, manifest, runtime binding, and tests requires manual
work. Matching uses the imported IR; adding an operation does not require a
matcher for that operation. Commit compiler extensions in Polygeist and update
this repository's submodule pin when distributing them.
