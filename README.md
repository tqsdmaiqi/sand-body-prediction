# Sand body thickness prediction code

Code accompanying _Reliable virtual sample augmentation with geological constraints for seismic prediction of sand body thickness_.

## Data availability

The data repository is available at [sand-body-data](https://github.com/tqsdmaiqi/sand-body-data). Place the seismic attribute file at `data/seismic_grid.csv` before running the prediction workflow.

The original study data are subject to confidentiality restrictions. Numerical values in the publicly released dataset have therefore been obfuscated to protect confidential information. Predictions, evaluation metrics and visualizations generated using this dataset are not guaranteed to match the results reported in the paper. The released data are intended to support workflow execution and methodological understanding rather than exact numerical reproduction of the original study.

## Prediction workflow

The verified compact workflow fits a boosted regression tree (BRT) to measured well data, predicts sand body thickness on a target grid, applies geological channel constraints and exports two Petrel scattered-data files. This workflow does not include virtual sample augmentation or independent validation and therefore does not reproduce the complete experimental protocol of the paper.

The verified Python interpreter is `D:\Users\yiwenWang\miniconda3\envs\DP-learn\python.exe`. Package dependencies are listed in `requirements.txt`. Run the following command from the project directory in PowerShell:

```powershell
& 'D:\Users\yiwenWang\miniconda3\envs\DP-learn\python.exe' predict.py
```

The workflow uses the following inputs:

- `data/integrated_raw.csv`: measured well coordinates, seismic attributes and observed sand body thickness.
- `data/seismic_grid.csv`: seismic attributes converted from the original Petrel exports. This project-relative path is the default input; an alternative file can be supplied using `--attributes PATH`.
- `data/prediction_grid_template.txt`: target coordinates and grid indices. Thickness values in this template are ignored.
- `data/gis/research.gpkg`: channel geometries; the layer name is specified in `config.toml`.

The model uses total energy, average energy, minimum amplitude, RMS amplitude and amplitude standard deviation. Seismic CSV coordinates must be named `X` and `Y`. All spatial inputs must use a consistent coordinate reference system with coordinates expressed in metres. BRT hyperparameters are specified in the `[brt]` section of `config.toml`.

Existing finite attribute values, including zeros, are retained at coincident locations. Missing target attributes are estimated separately using inverse-distance-squared weighting of the eight nearest valid observations. The workflow reports interpolation counts and maximum nearest-neighbour distances. All 38,609 template points are retained; some locations in the current dataset require estimates from observations approximately 2 km away.

The following files are written to `outputs`, replacing existing files with the same names:

- `coarse_grid_sand_thickness_scattered_data`: unconstrained BRT predictions.
- `scattered_sand_thickness_channel_constrained`: predictions after channel soft constraints.

Both files retain the Petrel scattered-data header and five data fields: `x`, `y`, `z`, `column` and `row`, where `z` denotes thickness in metres. The generated files have been checked for matching coordinates and grid indices, unique grid indices, finite values and non-negative thicknesses. These checks establish output integrity rather than predictive accuracy.

## Visualization

```powershell
& 'D:\Users\yiwenWang\miniconda3\envs\DP-learn\python.exe' visualize_results.py
```

The script exports a side-by-side comparison to `figures/prediction_comparison.png`. Add `--show` to open the figure window. Both panels use a common colour scale of 4–18 m. Values below 4 m are displayed as zero in white, and values above 18 m are capped at 18 m. Grey indicates locations absent from the target grid. Axis coordinates are expressed in kilometres. These transformations affect visualization only; the prediction files remain unchanged.

## Petrel conversion

Existing seismic CSV data can be used directly. To regenerate a CSV from the original attribute exports:

```powershell
& 'D:\Users\yiwenWang\miniconda3\envs\DP-learn\python.exe' petrel_to_csv.py '<Petrel attribute directory>' 'data\seismic_grid.csv'
```

Supply the resulting file using `predict.py --attributes 'data\seismic_grid.csv'`. The conversion step was not rerun during verification of the current prediction outputs.

## Modules

- predict.py: compact end-to-end prediction and Petrel scattered-data export.
- visualize_results.py: comparison of the two prediction maps using a shared display scale.
- main.py: task dispatcher; `python main.py <task>` executes one configured task at a time.
- config.toml / config.py: paths, parameters and configuration loading.
- dataset.py: CSV reading with explicitly supplied options.
- metrics.py: R2, RMSE and MAE computation.
- petrel.py: conversion of Petrel attribute exports into one grid CSV through the `petrel` task.
- petrel_to_csv.py: standalone Petrel attribute converter.
- features.py: feature ranking.
- sampling.py: sample preparation and virtual sample generation.
- sample_statistics.py: sample statistics.
- train.py: BRT fitting and evaluation.
- channel.py: channel features, geological constraints and model calculations.
- structure.py / evaluate.py: structural and predictive evaluation.
- ablation.py: two-stage cross-validation experiments.
- uncertainty.py: uncertainty statistics.
- reproduce.py: raw-well-data trial with a relative-deviation check.
- plotting.py / plot_map.py / plot_validation.py: additional figure generation.

Tasks dispatched through `main.py` require their respective configured input and intermediate files. They are separate from the compact workflow verified above.
