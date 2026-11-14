"""Generate two Petrel scattered-data maps from measured wells and seismic attributes."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import GradientBoostingRegressor
from config import load_config
from channel import (
    load_channel_data,
    calculate_channel_distance,
    calculate_channel_weights_for_points,
    build_channel_targets,
    apply_channel_soft_constraint,
)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attributes",
        type=Path,
        default=root / "data" / "seismic_grid.csv",
    )
    args = parser.parse_args()
    features = [
        "total_energy",
        "average_energy",
        "minimum_amplitude",
        "rms_amplitude",
        "std_amplitude",
    ]
    template = root / "data/prediction_grid_template.txt"
    grid = pd.read_csv(
        template,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["X_center", "Y_center", "reference_z", "column", "row"],
    )
    wells = pd.read_csv(root / "data/integrated_raw.csv")
    attributes = pd.read_csv(args.attributes, usecols=["X", "Y", *features])
    attributes = attributes.replace([np.inf, -np.inf], np.nan)
    if attributes[["X", "Y"]].isna().any().any():
        raise ValueError("Invalid seismic coordinates")
    coordinates = attributes[["X", "Y"]].to_numpy()
    query = grid[["X_center", "Y_center"]].to_numpy()
    prediction_features = np.empty((len(grid), len(features)))
    for j, feature in enumerate(features):
        values = attributes[feature].to_numpy()
        valid = np.isfinite(values)
        if not valid.any():
            raise ValueError(f"No valid values for {feature}")
        distances, indices = cKDTree(coordinates[valid]).query(
            query, k=min(8, valid.sum())
        )
        weights = 1 / np.maximum(distances, 1e-6) ** 2
        interpolated = (values[valid][indices] * weights).sum(axis=1) / weights.sum(
            axis=1
        )
        exact = distances[:, 0] < 1e-6
        interpolated[exact] = values[valid][indices[exact, 0]]
        prediction_features[:, j] = interpolated
        print(
            f"{feature}: {int((~valid).sum())} missing source values; {int((~exact).sum())} grid points interpolated; maximum nearest distance {distances[:, 0].max():.2f} m",
            flush=True,
        )
    model = GradientBoostingRegressor(**load_config("train")["brt_params"])
    model.fit(wells[features].to_numpy(), wells["sand_thickness"])
    grid["prediction"] = np.maximum(model.predict(prediction_features), 0)
    params = load_config("channel")
    channels = load_channel_data(root / "data/gis/research.gpkg", params["gis_layer"])
    if not channels.valid:
        raise ValueError("No valid channel polygons")
    well_geo = calculate_channel_distance(
        wells.rename(columns={"X": "X_center", "Y": "Y_center"}),
        channels,
        params=params,
    )
    targets = build_channel_targets(
        wells.sand_thickness.to_numpy(),
        well_geo.channel_level.to_numpy(),
        min_samples=params["min_target_samples"],
        params=params,
    )
    result = calculate_channel_distance(grid, channels, params=params)
    result = calculate_channel_weights_for_points(
        result,
        params["channel_bandwidth"],
        decay_mode=params["decay_mode"],
        params=params,
    )
    result = apply_channel_soft_constraint(result, "prediction", targets, params=params)
    header = "".join(
        line
        for line in template.read_text().splitlines(keepends=True)
        if line.startswith("#")
    )
    for name, column in [
        ("coarse_grid_sand_thickness_scattered_data", "prediction"),
        (
            "scattered_sand_thickness_channel_constrained",
            "pred_sand_thickness_channel_constrained",
        ),
    ]:
        values = result[["X_center", "Y_center", column, "column", "row"]].to_numpy()
        if not np.isfinite(values).all() or (values[:, 2] < 0).any():
            raise ValueError("Invalid prediction values")
        output = root / "outputs" / name
        temporary = output.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(header)
            np.savetxt(stream, values, fmt=["%.6f", "%.6f", "%.6f", "%d", "%d"])
        temporary.replace(output)
        print(
            f"Saved {output.name}: {len(values)} points, thickness {values[:, 2].min():.3f}–{values[:, 2].max():.3f} m",
            flush=True,
        )


if __name__ == "__main__":
    main()
