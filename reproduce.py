import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import GradientBoostingRegressor
from dataset import read_csv
from sampling import prepare_samples, generate_samples
from train import make_arrays
from metrics import regression_metrics
from channel import (load_channel_data, calculate_channel_distance,
                     calculate_channel_weights_for_points, build_channel_targets,
                     apply_channel_soft_constraint)


def run(*, params):
    sp, tp, cp = params["sampling"], params["training"], params["channel"]
    source = pd.read_csv(params["source_csv"], encoding="utf-8-sig")
    wells = source.copy()
    wells.columns = ["X_center", "Y_center", *sp["feature_cols"], "sand_thickness"]
    wells["source_well"] = [f"row_{i}" for i in range(len(wells))]
    grid = read_csv(sp["grid_attr_csv"], options=sp["csv_options"])
    channels = read_csv(sp["grid_channel_csv"], options=sp["csv_options"])
    _, indices = cKDTree(grid[["X_center", "Y_center"]]).query(wells[["X_center", "Y_center"]])
    wells[["grid_row", "grid_col"]] = grid.iloc[indices][["grid_row", "grid_col"]].to_numpy()
    validation = wells.iloc[params["validation_rows"]].copy()
    real = wells.drop(index=params["validation_rows"]).reset_index(drop=True)
    prepared, candidates = prepare_samples(real, grid, channels, params=sp)
    samples = generate_samples(prepared, candidates, params=sp)
    rename = dict(zip(cp["grid_attr_map"].values(), tp["feature_cols"]))
    rename["sand_thickness"] = tp["target_col"]
    samples = samples.rename(columns=rename)
    validation = validation.rename(columns=rename)
    tx, vx, ty, vy = make_arrays(samples, validation, params=tp)
    model = GradientBoostingRegressor(**tp["brt_params"])
    model.fit(tx, ty, sample_weight=samples["sample_weight"])
    geometry = load_channel_data(cp["gis_database"], cp["gis_layer"])
    real_geo = calculate_channel_distance(real, geometry, params=cp)
    targets = build_channel_targets(real["sand_thickness"].to_numpy(),
        real_geo["channel_level"].to_numpy(), min_samples=cp["min_target_samples"], params=cp)
    tables, rows = {}, []
    paper = {("train", "before"): (0.95, 1.48, 1.18),
             ("validation", "before"): (0.77, 3.10, 2.46),
             ("train", "after"): (0.96, 1.30, 1.04),
             ("validation", "after"): (0.88, 2.07, 1.63)}
    for name, frame, x, y in [("train", samples, tx, ty), ("validation", validation, vx, vy)]:
        predicted = np.maximum(model.predict(x), 0)
        points = calculate_channel_distance(frame.copy(), geometry, params=cp)
        points = calculate_channel_weights_for_points(points, cp["channel_bandwidth"],
            decay_mode=cp["decay_mode"], params=cp)
        points["pred_sand_thickness_raw"] = predicted
        points = apply_channel_soft_constraint(points, "pred_sand_thickness_raw", targets, params=cp)
        adjusted = points["pred_sand_thickness_channel_constrained"].to_numpy()
        tables[name] = pd.DataFrame({"source_well": frame["source_well"].to_numpy(),
            "observed": y, "before": predicted, "after": adjusted})
        for stage, values in [("before", predicted), ("after", adjusted)]:
            actual = regression_metrics(y, values)
            for metric, target in zip(("R2", "RMSE", "MAE"), paper[name, stage]):
                deviation = abs(actual[metric] - target) / abs(target)
                rows.append({"dataset": name, "stage": stage, "metric": metric,
                    "paper": target, "actual": actual[metric],
                    "relative_deviation": deviation, "within_tolerance": deviation <= params["tolerance"]})
    output = params["out_dir"]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "metric_comparison.csv", index=False)
    samples.to_csv(output / "generated_training.csv", index=False)
    for name, table in tables.items():
        table.to_csv(output / f"{name}_predictions.csv", index=False)
