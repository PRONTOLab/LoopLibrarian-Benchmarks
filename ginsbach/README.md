# Ginsbach/IDL comparison

Run the authors' IDL compiler on all 30 PolyBench programs, then execute the
programs containing detected replacements on the x86 host.

```
python3 setup.py ginsbach
python3 reproduce.py ginsbach --output results/runs/idl
```

Allow 10–30 minutes after [setup](../docs/reproduction.md). The expected result
is four detected idiom instances in three programs: one in correlation, two in
covariance, and one in trmm. All three programs fail the full-output comparison;
reproducing these failures is the reported result, not a successful acceleration.
`summary.json` records detections and execution checks.

The driver uses the original PolyBench allocation helpers from
`inputs/polybench_original_utilities/`; changing these helpers affects IDL's
detection. The compiler compatibility patches are included in this directory.
