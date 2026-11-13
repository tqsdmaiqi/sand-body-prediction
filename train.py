from metrics import regression_metrics as calc_metrics
from dataset import read_csv
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


def make_arrays(train_df: pd.DataFrame, valid_df: pd.DataFrame, *, params):
    train_x_raw = train_df[params["feature_cols"]].apply(pd.to_numeric, errors="coerce")
    valid_x_raw = valid_df[params["feature_cols"]].apply(pd.to_numeric, errors="coerce")
    med = train_x_raw.median(numeric_only=True)
    train_x = train_x_raw.fillna(med).fillna(0.0).to_numpy(dtype=float)
    valid_x = valid_x_raw.fillna(med).fillna(0.0).to_numpy(dtype=float)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    valid_x = scaler.transform(valid_x)
    train_y = pd.to_numeric(train_df[params["target_col"]], errors="coerce").to_numpy(dtype=float)
    valid_y = pd.to_numeric(valid_df[params["target_col"]], errors="coerce").to_numpy(dtype=float)
    return (train_x, valid_x, train_y, valid_y)


def fit_evaluate(train_df, valid_df, *, params):
    train_df = train_df.dropna(subset=[params["target_col"]]).reset_index(drop=True)
    valid_df = valid_df.dropna(subset=[params["target_col"]]).reset_index(drop=True)
    train_x, valid_x, train_y, valid_y = make_arrays(train_df, valid_df, params=params)
    sample_weight = None
    if params["sample_weight_col"] in train_df.columns:
        sample_weight = (
            pd.to_numeric(train_df[params["sample_weight_col"]], errors="coerce")
            .fillna(1.0)
            .clip(lower=0.0)
            .to_numpy(dtype=float)
        )
    model = GradientBoostingRegressor(**params["brt_params"])
    model.fit(train_x, train_y, sample_weight=sample_weight)
    train_pred = np.clip(model.predict(train_x), 0.0, None)
    valid_pred = np.clip(model.predict(valid_x), 0.0, None)
    metrics = pd.DataFrame(
        [
            {"dataset": "train_fullfit", **calc_metrics(train_y, train_pred)},
            {"dataset": "independent_validation", **calc_metrics(valid_y, valid_pred)},
        ]
    )
    predictions = pd.DataFrame({"y_true": valid_y, "y_pred": valid_pred})
    return (metrics, predictions)


def run(*, params) -> None:
    params["out_dir"].mkdir(parents=True, exist_ok=True)
    train_df = read_csv(params["train_csv"], options=params["csv_options"])
    valid_df = read_csv(params["valid_csv"], options=params["csv_options"])
    metrics, predictions = fit_evaluate(train_df, valid_df, params=params)
    metrics.to_csv(
        params["out_dir"] / "04_validation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        params["out_dir"] / "04_validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
