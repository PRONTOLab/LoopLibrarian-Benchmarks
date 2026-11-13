# Compilation cost

Measure fresh C-to-linked-AArch64 builds sequentially, with five repetitions
per input in each of two arms: Egglog rewrites enabled and disabled. Source,
compiler, configuration, executable hashes, commands, stage timings, peak RSS,
proof time, graph sizes, iteration counts, selected library calls, and timeout
outcomes are retained. The proof deadline is ten seconds, enforced with an
isolated process; process and instrumentation overhead remain in the measured
wall time. Timed-out proofs cannot authorize library calls.

`compilation-table.csv` contains suite-level mean, median, minimum and maximum
of complete per-input median build times. `measurement-tables.csv` provides
the same aggregation for stage times, proof measurements and process memory;
each metric retains its own complete/missing denominator. `summary.json`
also retains per-input quartiles, call selections and failed repetitions.

```
python3 reproduce.py compilation_cost --output results/runs/compilation-cost
```

`inputs.json` contains 199 distinct retained build inputs: 30 PolyBench,
23 Llama, 115 ATen, and 31 MFEM. The Llama compilation cohort includes its
19 operation fixtures, extended forward, and three public llama2.c functions;
it is not the 19-case coverage denominator. ATen uses the recorded paper shapes
and matching configuration in both arms. Failed builds and unmatched programs
are retained. A table entry uses only inputs with all requested repetitions
successfully linked, and reports excluded counts.

The published table used three repetitions. `first-three-compilation-table.csv`
summarizes medians from repetitions 1, 2, and 3, requiring all three to succeed.
Its denominator is separate from the five-repetition table, so a later failed
build cannot silently remove an input from the three-repetition subset.
The summary retains both sets of medians; neither should be described as a
reproduction of the old table until the cohort and measurement boundaries have
been reconciled. This driver includes compilation, runtime-object generation,
and final linking. Earlier matching-only or no-link probes are not equivalent.

A full run is 1,990 builds. Initial full-build probes took roughly 50–170 seconds
each on the shared host, suggesting approximately 30–60 hours for the complete
sequential experiment. This estimate is provisional; it excludes toolchain
setup and can grow when matching reaches its time limits. A short preflight is:

```
python3 reproduce.py compilation_cost --cases aten/aten_avg_pool2d --repetitions 1 --output results/runs/compilation-preflight
```

Use the suite driver directly with `--resume` to continue a stopped campaign;
it requires identical compiler inputs, benchmark configuration, and sources.
Do not rebuild or edit the compiler during a campaign. Use `--stage summarize`
to inspect incomplete progress; it uses the saved cohort and repetition count,
without requiring the original compiler or input files. Application compiler-result reuse is disabled;
a campaign-local parsed reference-catalog cache is reused separately per arm.
GNU time's peak RSS is the maximum of a process, not the sum of simultaneously
resident parent/worker processes. Graph records retain the counting settings
used by each proof engine.
