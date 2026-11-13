# Sand body thickness prediction code

Code accompanying _Reliable virtual sample augmentation with geological constraints for seismic prediction of sand body thickness_.

## Data availability

The data used in this study are confidential. Any data released publicly will be desensitized before publication. Desensitization may change values, spatial positions or sample composition, so metrics computed on the public data may not match the values reported in the paper; this is expected and not an error. The public data are released to let readers run the complete workflow and understand the method and the data handoff between modules, not to reproduce the paper's exact numeric results.

## Modules

- main.py: single entry point; `python main.py <task>` dispatches one task at a time
- config.toml / config.py: paths, parameters and configuration loading
- dataset.py: CSV reading with explicitly supplied options
- metrics.py: R2, RMSE and MAE computation
- petrel.py: convert Petrel attribute exports into one grid CSV (the `petrel` task)
- petrel_to_csv.py: standalone converter for a folder of Petrel attribute files; usage `python petrel_to_csv.py <source_dir> <output_csv>`
- features.py: feature ranking
- sampling.py: sample preparation and virtual sample generation
- sample_statistics.py: sample statistics
- train.py: BRT fitting and evaluation
- channel.py: channel features, constraints and model calculations
- structure.py / evaluate.py: structural and effect evaluation
- ablation.py: two-stage cross-validation experiments
- uncertainty.py: uncertainty statistics
- reproduce.py: raw-well-data trial with the relative-deviation check
- plotting.py / plot_map.py / plot_validation.py: figure generation
