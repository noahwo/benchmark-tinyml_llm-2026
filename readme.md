# Benchmark TinyML LLM 2026

> MCU referes to dp, mc, sg, and MPU refers to psg and tpusg. 

This repo collects experiment outputs, similarity scoring code, and plotting notebooks only necessary to plot:

#### For MCU:
- G4o
- G4o-m
- Qw14
- Qw32
- Phi4
- Co22
- Co22-p (only one with a series tested with parameters for MCU)
#### For MPU, both w/ and w/o parameters:
- Co22
- Ge3 (gemma3)
- G5
- Phi4
- Qw14
- Qw32
#### For ablation (for-abla), targeting only MPU
- G5
- Phi4
- Qw32

## Quick Map
- `archive` — Forget about this, some messy plotting using dataset from the middleware paper.
- `codebertscore-similarity/` — CodeBERTScore tooling and results.
	- `analyser-results/` — Notebooks to calculate scores and simply visualiza some results. And similarity result CSVs.
	- `references/` — Three reference files (sketches) to calculate against. 
	- others - Leave them as they are
- `langfuse_export/` — Trace exporting and pre-processinf only relevant to ablation and newest necessary tests. The similarity calculation directly reads code files here.
- `plot-exploration/for-*` - The `.csv` datasets ready for plotting, the content is listed above. 

## Note
- `convert-csv.ipynb` accepts CSVs from both right after langfuse export (under `langfuse_export`) and the CSVs used from the middleware paper (extracted from those Excel files), and converts to current CSV files in this repo. Only re-orders columns, expands/merges/removes some columns. 

 