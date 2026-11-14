# Additional ATen matching experiment

This experiment investigates fixtures outside both the 115-case paper timing
cohort and the 268-case pointwise inventory. It does not change either denominator.
The initial fixed list is `additional22.csv`; `additional_gather9.csv` extends
that search with nine more candidates. `additional_candidates.csv` preserves
all 31 candidates, including cases that remain partial. A final
nearest-exact 2D candidate is frozen in `additional_nearest_exact2d.csv`;
`additional_all_candidates.csv` is the 32-case union. Each list records the
original C source hash. No application fixture is edited.

## Recorded result

**22 additional cases have complete strict matching records out of 32 tested.**
Twelve use the new gather imports; ten were handled by existing arithmetic or
copy imports and are newly verified here. This is a coverage result for the
original fixtures, not a GPU correctness or performance result. It leaves the
separate pointwise result at 244/268 and does not alter the paper's 115-case
cohort. The saved campaign aggregate used for selection contained none of these
32 cases; this experiment does not claim a fresh scan of all 598 fixtures.

All 32 source hashes remain unchanged. Every complete case has verified proof,
successful CUDA ABI validation, and zero remaining Linalg operations or loops.
The first full recheck finished with 21/31 complete; the separate nearest-exact
2D run supplied the twenty-second. Both used the same compiler and catalog.
Per-case results, emitted specifications, hashes and evidence paths are in
[`additional_results.csv`](additional_results.csv). Full logs and proofs are
under `results/runs/aten-additional-candidates/matching/` and
`results/runs/aten-additional-nearest-exact2d/matching/`.

The ten incomplete cases remain recorded:

- `xlogy` and `xlog1py`: guarded memory reads remain before tensor matching.
- `fill_diagonal_cpu`: the diagonal output view lacks a complete matching proof.
- Six reflection/replication padding fixtures: coordinate selection and
  computed-index gathers remain partly unmatched.
- `sparse_dense_intersection_cpu`: multiplication matches, but the gather's
  producer dataflow does not. Its imported coordinate-gather reference passes
  native and raising checks; that alone is not an application match.

The Thrust regression suite passes all seven tests. No new application-specific
compiler matcher or pass was added in this experiment.

## Reproduce matching

With the compiler and dependencies configured, rebuild the reference catalog:

```
export POLYGEIST_SPEC_SUITE=lib_v3
python3 third_party/Polygeist/scripts/correctness/lib_v2/build_lib_v2.py --incremental -j 6
python3 scripts/correctness/aten_paper115_campaign.py --checklist aten/additional_all_candidates.csv --output results/runs/aten-additional-candidates/matching --ranks 1 --region --split-pointwise-inputs --numeric-model strict --jobs 6
```

Use a fresh output directory. Despite its historical filename, the campaign
driver takes this explicit checklist; it does not substitute the paper cohort.
It raises the C fixtures, verifies imported-reference proofs, and checks the
CUDA ABI. A complete match requires emitted calls and no residual Linalg
computation or loops. Matching evidence is separate from GPU execution and
performance. Allow approximately 5–15 minutes for the 32-case matching batch
with an existing compiler, depending on host load and reference-cache state.

## Library imports

Seven typed Thrust gather references cover flat indices, selected rows or
columns, per-element column indices, one column per row, and independent
row/column coordinates. Row selection supports both float and integer data.
They expand public Thrust's permutation-iterator traversal into C and are
raised through the shared pipeline. Runtime adapters construct typed index
iterators and call actual `thrust::gather`; the compiler does not recognize
application names or contain a gather-specific matching pass.

Five additional public Thrust integer arithmetic and bound-selection imports
expose operations used in padding coordinates. They preserve unsigned
wraparound and signed comparisons. These partial padding matches are retained
in the results and are not counted as complete.

Native checks compare the reference implementations with public Thrust over
189 gather configurations, including repeated/nonmonotonic indices, NaNs,
signed zero, integer extremes, unchanged inputs and output guards. Integer
coordinate checks cover 203 boundary-element comparisons. All twelve new imports
pass exact reference comparisons at the affine, Linalg and production tensor
stages. The updated runtime compiles for sm_87. This experiment adds no GPU
correctness passes or timings.

```
python3 -m unittest discover -s third_party/Polygeist/scripts/correctness/tests -p 'test_thrust_*.py'
```
