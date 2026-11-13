from pathlib import Path
import tomllib


def load_config(task):
    with Path(__file__).with_name("config.toml").open("rb") as file:
        settings = tomllib.load(file)
    params = dict(settings[task])
    for key in (
        "input_csv",
        "train_csv",
        "valid_csv",
        "predict_csv",
        "channel_grid_csv",
        "grid_attr_csv",
        "real_csv",
        "grid_channel_csv",
        "virtual_csv",
        "out_dir",
        "local_train_csv",
        "out",
        "src",
        "ref",
        "pred_csv",
        "label_csv",
    ):
        if key in params:
            params[key] = Path(params[key])
    for key in (
        "zone_name",
        "channel_bandwidth",
        "bandwidth_search_space",
        "channel_target_quantile",
        "channel_level_strength",
        "keep_ratio",
        "base_weight",
    ):
        if key in params:
            params[key] = dict(params[key])
    params["csv_options"] = settings["dataset"]["csv"]
    params["output_dir"] = Path(settings["paths"]["output_dir"])
    params["channel_output"] = Path(settings["paths"]["channel_output_dir"])
    if task in ("train", "channel", "ablation"):
        section = {
            "train": "brt",
            "channel": "channel_brt",
            "ablation": "ablation_brt",
        }[task]
        params["brt_params"] = settings[section]
    if task == "channel":
        params["gis_database"] = Path(settings["gis"]["database"])
        params["gis_layer"] = settings["gis"]["channel_layer"]
        params["grid_nrow"], params["grid_ncol"] = params["grid_shape"]
    if task == "structure":
        params["row_n"], params["col_n"] = params["grid_shape"]
    if task == "plot_validation":
        params["pred_col"], params["kind_lbl"] = params["pred_cols"][
            params["pred_kind"]
        ]
    if task == "reproduce":
        params["sampling"] = load_config("sampling")
        params["training"] = load_config("train")
        params["channel"] = load_config("channel")
        params["source_csv"] = Path(params["source_csv"])
    return params
