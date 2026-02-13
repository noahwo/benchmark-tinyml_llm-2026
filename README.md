# Benchmark TinyML LLM 2026

> MCU referes to dp, mc, sg, and MPU refers to psg and tpusg. 

This repo collects experiment outputs, similarity scoring code, and plotting notebooks only necessary to plot:

> NOTE: `gpt-5_sg_converted.csv` so far has never been processed or plotted yet, did not have enough time :(


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
- `plottings/` - I created the statistics CSVs directly usable for plotting, so no calculation will be needed, just read the values and use them to plot. **Please please see [this plottings_README.md](plottings_README.md).**
- `plot-exploration/` - Categorized by MPU, MCU, and Ablation: exported CSV files, merged one-CSV file, notebooks for explorationarily plotting, and output figs. 
	- `for-abla/` - For ablation
		- `figs/` - Generated figures
		- `seperated_csv/` - Datasets CSV by models, or processors, or ablation levels.
		- `all-abla.csv` I merged all the csv files for ablation, this is it. Easier to read.
		- `box-plots_ablation.ipynb` - Box plots similar to the middleware paper
		- `success-rate_ablation.ipynb` - Plots success rate bar chart, and cost & time comsumption by GPT models, like in the Middleware paper.
	- `for-MCU/` - Same as above. But the `all-MCU2.csv` differs from `all-MCU.csv` only that I put `codestral-p` as a seperated model, for easier plotting. 
	- `for-MPU/` - Same as above.
- `codebertscore-similarity/` — CodeBERTScore tooling and results.
	- `analyser-results/` — Notebooks to calculate scores and simply visualiza some results. And similarity result CSVs.
	- `references/` — Three reference files (sketches) to calculate against. 
	- others - Leave them as they are
- `langfuse_export/` — As the name tells. Traces, generated code, csv extracting etc.





# Unimportant
- `convert-csv.ipynb` accepts CSVs from both right after langfuse export (under `langfuse_export`) and the CSVs used from the middleware paper (extracted from those Excel files), and converts to current CSV files in this repo. Only re-orders columns, expands/merges/removes some columns. 

 