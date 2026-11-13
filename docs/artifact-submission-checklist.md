# Artifact submission checklist

Reviewed against the
[cTuning author checklist](https://github.com/ctuning/artifact-evaluation/blob/master/docs/checklist.md)
and the [CGO 2027 call](https://2027.cgo.org/track/cgo-2027-artifact-evaluation).
The former guides artifact descriptions; use the latter for conference-specific
submission and badge requirements. Apply the relevant checklist items to this
artifact. Collective Mind integration is optional.

This tracks submission preparation, not benchmark completion. A checked item
means the named documentation exists; it does not certify a clean installation,
successful independent reproduction, or an awarded badge.

## Documentation already present

- [x] Testing hardware, software versions, measurement controls, and differences
  from the historical environment: [README](../README.md#testing-systems) and
  [machine-readable specifications](system-specifications.json).
- [x] Setup and experiment entry points, expected output files, and correctness
  gates for the eight established drivers:
  [reproduction instructions](reproduction.md). Other suites have their own
  READMEs; this does not establish complete coverage of every paper result.
- [x] Approximate preparation and experiment durations:
  [time estimates](../README.md#expected-experiment-time). Estimates are labeled
  and still need confirmation from a clean release run.
- [x] Customization examples and the manual work needed for new benchmarks,
  references, and executable library bindings:
  [extension guide](extending-experiments.md).
- [x] Repository license and upstream-source notices:
  [LICENSE](../LICENSE) and [source notices](SOURCE_NOTICES.md).

## Work to close before handing the artifact to reviewers

- [ ] **Submission document.** Prepare the accepted paper with the artifact
  appendix, limited to two pages and placed before References. Its abstract
  should identify LoopLibrarian's supported claims and the required host,
  execution target, and separately installed dependencies. Link this repository
  at the exact evaluated revision.
- [ ] **Frozen release.** Select the benchmark and compiler revisions, verify
  all submodule pins and downloaded-source hashes, and package the intended
  files. Confirm that licenses and acquisition instructions cover each shipped
  or externally supplied component. A branch URL alone does not identify the
  tested release.
- [ ] **Clean installation and quick check.** Exercise the instructions from a
  fresh checkout of those revisions. Provide a short host check and an Orin
  check with commands, expected output, elapsed time, and saved logs. The
  extension guide's command-plan checks have not established this.
- [ ] **Container.** Finish and test the full-artifact container/build recipe
  and launch instructions. The current SpEQ-specific Dockerfile does not
  package LoopLibrarian. Explain the x86 build environment and separate Orin
  execution environment. CGO encourages containers; this is our packaging
  task rather than a mandatory conference format.
- [ ] **Resource budget.** Measure download size, unpacked sources, build and
  container storage, result storage, and peak memory. Separate preparation
  costs from experiment costs and state the retained-output assumptions.
- [ ] **Reviewer access.** Document obtaining the target libraries and any PVA
  SDK, required privileges for installation or clock control, and the execution
  access arrangement. Determine whether the hardware/proprietary dependency
  setup needs advance coordination with the chairs. Keep credentials outside
  the artifact.
- [ ] **Input inventory.** Identify each benchmark, dataset, and model fixture
  as included, generated, or downloaded; record its version, size, provenance,
  and setup command. Explain the exact Llama fixture scope and any difference
  from running a full pretrained model.
- [ ] **Claim-to-result index.** Map every submitted figure/table claim to its
  case cohort, command, output file, comparison baseline, and expected outcome.
  Include regeneration/plotting commands and clearly identify negative results
  and unsupported cases. Existing suite descriptions are a starting point.
- [ ] **Performance interpretation.** Complete the per-experiment guidance on
  expected variation using measurements and hardware differences. Preserve
  correctness tolerances separately; do not invent a universal percentage or
  adjust correctness gates to obtain a timing match.
- [ ] **Measured walkthrough.** Record actual setup and suite durations from
  the frozen release and reconcile them with the README's planning estimates.
- [ ] **Reuse demonstration.** Run a documented extension, such as an alternate
  PolyBench size, and retain its output. Check that the library-import example
  is reproducible with the pinned dependencies and stated manual steps.
- [ ] **Archival deposit.** Prepare a versioned archive and DOI for the
  Artifacts Available badge. GitHub is the development location; it does not
  replace a qualifying archival deposit. Record how the archive preserves or
  retrieves pinned dependency sources.

## Badge interpretation

Functional and Reusable are alternative levels of Artifacts Evaluated; Reusable
requires more than functional scripts. Results validation requires independent
evaluation, and availability concerns archival access. Describe these as targets
until the corresponding evaluation or deposit establishes eligibility. See the
[CGO badge definitions](https://2027.cgo.org/track/cgo-2027-artifact-evaluation)
for the governing criteria.
