from dataset import read_csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def normalize_channel_fields(df: pd.DataFrame, *, params) -> pd.DataFrame:
    out = df.copy()
    for col in params["channel_cols"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["distance_to_channel"] = out["distance_to_channel"].fillna(np.inf)
    out["channel_weight_2000"] = out["channel_weight_2000"].fillna(0.0).clip(0.0, 1.0)
    out["in_channel_3000"] = out["in_channel_3000"].fillna(0.0).clip(0.0, 1.0)
    out["channel_zone"] = out["channel_zone"].fillna(0.0)
    return out


def prepare_samples(
    real, grid_attr, grid_channel, *, params
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = grid_attr.merge(
        grid_channel[["grid_row", "grid_col", *params["channel_cols"]]],
        on=["grid_row", "grid_col"],
        how="left",
    )
    grid = normalize_channel_fields(grid, params=params)
    real = real.merge(
        grid[["grid_row", "grid_col", *params["channel_cols"]]],
        on=["grid_row", "grid_col"],
        how="left",
    )
    real = normalize_channel_fields(real, params=params)
    real = real.dropna(
        subset=[params["target_col"], "grid_row", "grid_col", "X_center", "Y_center"]
    ).reset_index(drop=True)
    grid = grid.dropna(
        subset=["grid_row", "grid_col", "X_center", "Y_center"]
    ).reset_index(drop=True)
    return (real, grid)


def make_scaled_features(real_df: pd.DataFrame, grid_df: pd.DataFrame, *, params):
    real_x = real_df[params["feature_cols"]].apply(pd.to_numeric, errors="coerce")
    grid_x = grid_df[params["feature_cols"]].apply(pd.to_numeric, errors="coerce")
    med = real_x.median(numeric_only=True)
    real_x = real_x.fillna(med).fillna(0.0)
    grid_x = grid_x.fillna(med).fillna(0.0)
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_x)
    grid_scaled = scaler.transform(grid_x)
    return (real_scaled, grid_scaled)


def candidate_ring(
    source_row: int, source_col: int, order: int
) -> set[tuple[int, int]]:
    cells = set()
    for dr in range(-order, order + 1):
        for dc in range(-order, order + 1):
            if dr == 0 and dc == 0:
                continue
            if max(abs(dr), abs(dc)) == order:
                cells.add((source_row + dr, source_col + dc))
    return cells


def calc_geo_weight(
    candidate: pd.Series, well_part: pd.DataFrame, *, params
) -> np.ndarray:
    cand_weight = float(candidate.get("channel_weight_2000", 0.0))
    cand_in = int(float(candidate.get("in_channel_3000", 0.0)) > 0)
    cand_zone = float(candidate.get("channel_zone", 0.0))
    well_weight = (
        pd.to_numeric(well_part["channel_weight_2000"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    well_in = (
        pd.to_numeric(well_part["in_channel_3000"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
        > 0
    ).astype(float)
    well_zone = (
        pd.to_numeric(well_part["channel_zone"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    strength_sim = np.exp(-(((cand_weight - well_weight) / params["sigma_geo"]) ** 2))
    in_sim = np.where(well_in == cand_in, 1.0, 0.7)
    zone_sim = np.where(np.abs(well_zone - cand_zone) < 1e-06, 1.0, 0.75)
    return strength_sim**0.5 * in_sim**0.3 * zone_sim**0.2


def generate_samples(real_df, grid_df, *, params):
    real_scaled, grid_scaled = make_scaled_features(real_df, grid_df, params=params)
    grid_key_to_idx = {
        (int(r), int(c)): i
        for i, (r, c) in enumerate(zip(grid_df["grid_row"], grid_df["grid_col"]))
    }
    real_xy = real_df[["X_center", "Y_center"]].to_numpy(dtype=float)
    real_y = pd.to_numeric(real_df[params["target_col"]], errors="coerce").to_numpy(dtype=float)
    raw_rows = []
    for source_idx, source in real_df.iterrows():
        source_row = int(source["grid_row"])
        source_col = int(source["grid_col"])
        source_feat = real_scaled[source_idx]
        for order in (1, 2):
            for key in candidate_ring(source_row, source_col, order):
                grid_idx = grid_key_to_idx.get(key)
                if grid_idx is None:
                    continue
                grid_row = grid_df.iloc[grid_idx]
                grid_feat = grid_scaled[grid_idx]
                grid_xy = grid_row[["X_center", "Y_center"]].to_numpy(dtype=float)
                attr_dist_to_source = float(np.linalg.norm(grid_feat - source_feat))
                space_dist_to_source = float(np.linalg.norm(grid_xy - real_xy[source_idx]))
                attr_dist_all = np.linalg.norm(real_scaled - grid_feat, axis=1)
                space_dist_all = np.linalg.norm(real_xy - grid_xy, axis=1)
                nearest = np.argsort(
                    attr_dist_all + space_dist_all / params["sigma_space"]
                )[: params["k_neighbor_wells"]]
                well_part = real_df.iloc[nearest].copy()
                sigma_attr = max(float(np.nanpercentile(attr_dist_all[nearest], 75)), 1e-06)
                attr_w = np.exp(-attr_dist_all[nearest] ** 2 / (2.0 * sigma_attr**2))
                space_w = np.exp(
                    -space_dist_all[nearest] ** 2 / (2.0 * params["sigma_space"] ** 2)
                )
                geo_w = calc_geo_weight(grid_row, well_part, params=params)
                joint = attr_w**0.5 * space_w**0.35 * geo_w**0.15
                if joint.sum() <= 1e-12:
                    continue
                s_attr = float(np.average(attr_w, weights=joint))
                s_space = float(np.average(space_w, weights=joint))
                s_geo = float(np.average(geo_w, weights=joint))
                virtual_confidence = 0.5 * s_attr + 0.3 * s_space + 0.2 * s_geo
                sand = float(np.sum(joint * real_y[nearest]) / np.sum(joint))
                sample_weight = float(
                    np.clip(
                        params["base_weight"][order] * virtual_confidence,
                        params["min_sample_weight"],
                        params["max_sample_weight"],
                    )
                )
                row = grid_row[
                    [
                        "X_center",
                        "Y_center",
                        "grid_row",
                        "grid_col",
                        *params["feature_cols"],
                        *params["channel_cols"],
                    ]
                ].to_dict()
                row.update(
                    {
                        "sand_thickness": max(sand, 0.0),
                        "sample_type": "virtual",
                        "sample_weight": sample_weight,
                        "source_well": str(source["source_well"]),
                        "neighbor_order": order,
                        "attr_dist": attr_dist_to_source,
                    }
                )
                raw_rows.append(row)
    virtual_df = pd.DataFrame(raw_rows)
    if virtual_df.empty:
        raise RuntimeError("No virtual samples generated; check well and grid index alignment.")
    kept = []
    for order, ratio in params["keep_ratio"].items():
        part = virtual_df[virtual_df["neighbor_order"].eq(order)].copy()
        cutoff = part["attr_dist"].quantile(ratio)
        kept.append(part[part["attr_dist"].le(cutoff)])
    virtual_df = pd.concat(kept, ignore_index=True)
    real_out = real_df.copy()
    real_out["sample_type"] = "real"
    real_out["sample_weight"] = 1.0
    real_out["neighbor_order"] = 0
    common_cols = [
        "X_center",
        "Y_center",
        "grid_row",
        "grid_col",
        "sand_thickness",
        "sample_type",
        "sample_weight",
        "source_well",
        "neighbor_order",
        *params["channel_cols"],
        *params["feature_cols"],
    ]
    train_df = pd.concat([real_out[common_cols], virtual_df[common_cols]], ignore_index=True)
    return train_df


def run(*, params) -> None:
    params["out_dir"].mkdir(parents=True, exist_ok=True)
    real = read_csv(params["real_csv"], options=params["csv_options"])
    grid_attr = read_csv(params["grid_attr_csv"], options=params["csv_options"])
    grid_channel = read_csv(params["grid_channel_csv"], options=params["csv_options"])
    real_df, grid_df = prepare_samples(real, grid_attr, grid_channel, params=params)
    train_df = generate_samples(real_df, grid_df, params=params)
    train_path = params["out_dir"] / "02_channel_training_samples.csv"
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
