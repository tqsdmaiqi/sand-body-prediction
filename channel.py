from metrics import regression_metrics as calc_metrics
from plotting import plot_scatter, plot_morphology
from pathlib import Path
from dataset import read_csv
import math
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from skimage.measure import label as cc_label
from skimage.morphology import skeletonize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


@dataclass
class ChannelRecord:
    geometry: Any
    area: float
    channel_level: int


@dataclass
class ChannelData:
    records: list[ChannelRecord]
    used_existing_level: bool
    p25: float | None
    p75: float | None
    backend: str

    @property
    def valid(self) -> bool:
        return len(self.records) > 0


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def validate_feature_columns(df: pd.DataFrame, name: str, *, params) -> None:
    missing = [
        c
        for c in [params["target_col"], *params["feature_cols"]]
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"{name} Missing columns: {missing}")


def make_arrays(train_df: pd.DataFrame, pred_df: pd.DataFrame, *, params):
    x_train_raw = train_df[params["feature_cols"]].apply(pd.to_numeric, errors="coerce")
    x_pred_raw = pred_df[params["feature_cols"]].apply(pd.to_numeric, errors="coerce")
    med = x_train_raw.median(numeric_only=True)
    x_train = x_train_raw.fillna(med).fillna(0.0).to_numpy(dtype=float)
    x_pred = x_pred_raw.fillna(med).fillna(0.0).to_numpy(dtype=float)
    scaler = StandardScaler()
    return (scaler.fit_transform(x_train), scaler.transform(x_pred))


def sample_weight(df: pd.DataFrame) -> np.ndarray | None:
    if "sample_weight" not in df.columns:
        return None
    sw = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).clip(0.0, 0.25)
    return sw.to_numpy(dtype=float)


def attach_channel_feature(
    df: pd.DataFrame, ch_df: pd.DataFrame, grid_df: pd.DataFrame, name: str, *, params
) -> pd.DataFrame:
    out = df.drop(columns=[*params["attr_cols"], "distance_to_channel"], errors="ignore")
    out = out.merge(grid_df, on=params["key_cols"], how="left")
    out = out.merge(
        ch_df[params["key_cols"] + ["distance_to_channel"]],
        on=params["key_cols"],
        how="left",
    )
    miss_a = int(out[params["attr_cols"]].isna().any(axis=1).sum())
    miss_d = int(out["distance_to_channel"].isna().sum())
    if miss_a or miss_d:
        out = out.dropna(subset=[*params["attr_cols"], "distance_to_channel"])
    out["distance_to_channel"] = pd.to_numeric(out["distance_to_channel"], errors="coerce")
    return out.reset_index(drop=True)


def classify_channel_by_area(
    area_values: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    area = np.asarray(area_values, dtype=float)
    p25 = float(np.nanpercentile(area, 25))
    p75 = float(np.nanpercentile(area, 75))
    levels = np.where(area >= p75, 1, np.where(area >= p25, 2, 3))
    return (levels.astype(int), p25, p75)


def _normalize_channel_level(value: Any) -> int | None:
    try:
        level = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return level if level in (1, 2, 3) else None


def load_channel_polygons_geopandas(database: Path, layer: str) -> ChannelData | None:
    import geopandas as gpd

    gdf = gpd.read_file(str(database), layer=layer)
    if gdf is None or len(gdf) == 0 or "geometry" not in gdf.columns:
        return None
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["channel_area"] = gdf.geometry.area.astype(float)
    gdf = gdf[gdf["channel_area"] > 0].copy()
    if len(gdf) == 0:
        return None
    has_level = "channel_level" in gdf.columns
    if has_level:
        levels = gdf["channel_level"].map(_normalize_channel_level)
        if levels.isna().any():
            auto_levels, p25, p75 = classify_channel_by_area(gdf["channel_area"].to_numpy())
            levels = levels.fillna(pd.Series(auto_levels, index=gdf.index))
        else:
            p25 = p75 = None
    else:
        levels, p25, p75 = classify_channel_by_area(gdf["channel_area"].to_numpy())
    records = [
        ChannelRecord(geometry=geom, area=float(area), channel_level=int(level))
        for geom, area, level in zip(gdf.geometry, gdf["channel_area"], levels)
    ]
    return ChannelData(records, has_level and p25 is None, p25, p75, "geopandas")


def load_channel_data(database: Path, layer: str) -> ChannelData:
    return load_channel_polygons_geopandas(database, layer)


def validate_bandwidths(bandwidth: dict[int, float]) -> None:
    if not bandwidth[1] >= bandwidth[2] >= bandwidth[3]:
        raise ValueError(f"Bandwidths must satisfy B1>=B2>=B3; received: {bandwidth}")


def _point_from_xy(x: float, y: float, backend: str):
    if backend == "arcpy":
        import arcpy

        return arcpy.PointGeometry(arcpy.Point(float(x), float(y)))
    from shapely.geometry import Point

    return Point(float(x), float(y))


def _distance(point: Any, geometry: Any, backend: str) -> float:
    if backend == "arcpy":
        if bool(geometry.contains(point)):
            return 0.0
        return float(point.distanceTo(geometry))
    if geometry.contains(point) or geometry.touches(point):
        return 0.0
    return float(point.distance(geometry))


def calculate_channel_distance(
    df: pd.DataFrame, channel_data: ChannelData, *, params
) -> pd.DataFrame:
    out = df.copy()
    for level in [1, 2, 3]:
        out[f"distance_to_channel_L{level}"] = np.nan
    out["distance_to_channel"] = np.nan
    if not channel_data.valid:
        out["channel_level"] = 0
        return out
    import shapely

    x = pd.to_numeric(out.get(params["x_col"]), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(out.get(params["y_col"]), errors="coerce").to_numpy(dtype=float)
    ok_xy = np.isfinite(x) & np.isfinite(y)
    points = shapely.points(np.where(ok_xy, x, 0.0), np.where(ok_xy, y, 0.0))
    records_by_level = {
        level: [r for r in channel_data.records if r.channel_level == level]
        for level in [1, 2, 3]
    }
    dist_by_level: dict[int, np.ndarray] = {}
    for level in [1, 2, 3]:
        recs = records_by_level[level]
        if not recs:
            dist_by_level[level] = np.full(len(out), np.nan)
            continue
        merged = shapely.union_all([r.geometry for r in recs])
        d = np.asarray(shapely.distance(points, merged), dtype=float)
        d[~ok_xy] = np.nan
        dist_by_level[level] = d
    stack = np.vstack([dist_by_level[1], dist_by_level[2], dist_by_level[3]])
    has_any = np.isfinite(stack).any(axis=0)
    with np.errstate(invalid="ignore"):
        idx = np.argmin(np.where(np.isfinite(stack), stack, np.inf), axis=0)
    nearest_level = np.where(has_any, idx + 1, 0).astype(int)
    nearest_dist = np.where(has_any, stack[idx, np.arange(stack.shape[1])], np.nan)
    for level in [1, 2, 3]:
        out[f"distance_to_channel_L{level}"] = dist_by_level[level]
    out["distance_to_channel"] = nearest_dist
    out["channel_level"] = nearest_level
    return out


def calculate_channel_weight(
    distance: float, level: int, bandwidth: dict[int, float], *, decay_mode, params
) -> float:
    if level not in (1, 2, 3) or not np.isfinite(distance):
        return 0.0
    b = float(bandwidth[level])
    if distance > b:
        return 0.0
    if decay_mode == "linear":
        w = max(0.0, 1.0 - distance / b)
    else:
        w = math.exp(-distance / b)
    return float(np.clip(w * params["channel_level_strength"][level], 0.0, 1.0))


def calculate_channel_weights_for_points(
    df: pd.DataFrame, bandwidth: dict[int, float], *, decay_mode, params
) -> pd.DataFrame:
    out = df.copy()
    validate_bandwidths(bandwidth)
    weights = []
    dominant_levels = []
    for _, row in out.iterrows():
        candidates = []
        for level in [1, 2, 3]:
            d = pd.to_numeric(row.get(f"distance_to_channel_L{level}"), errors="coerce")
            w = (
                calculate_channel_weight(
                    float(d), level, bandwidth, decay_mode=decay_mode, params=params
                )
                if np.isfinite(d)
                else 0.0
            )
            candidates.append((level, w))
        level_best, w_best = max(candidates, key=lambda x: x[1])
        if w_best <= 0:
            level_best = 0
        weights.append(w_best)
        dominant_levels.append(level_best)
    out["channel_level"] = dominant_levels
    out["channel_weight"] = weights
    return out


def build_channel_targets(
    y_train: np.ndarray, levels: np.ndarray | None = None, *, min_samples, params
) -> dict[int, float]:
    y = np.asarray(y_train, dtype=float)
    targets: dict[int, float] = {}
    for level, q in params["channel_target_quantile"].items():
        fallback = float(np.nanquantile(y, q))
        if levels is None:
            targets[level] = fallback
            continue
        sel = y[np.asarray(levels) == level]
        sel = sel[np.isfinite(sel)]
        targets[level] = float(np.median(sel)) if len(sel) >= min_samples else fallback
    return targets


def apply_channel_soft_constraint(
    df: pd.DataFrame,
    pred_col: str,
    channel_targets: dict[int, float],
    out_col: str = "pred_sand_thickness_channel_constrained",
    *,
    params,
) -> pd.DataFrame:
    out = df.copy()
    y_brt = pd.to_numeric(out[pred_col], errors="coerce").to_numpy(dtype=float)
    raw_weight = (
        pd.to_numeric(out.get("channel_weight", 0.0), errors="coerce")
        .fillna(0.0)
        .clip(0, 1)
        .to_numpy(dtype=float)
    )
    blend_weight = np.clip(raw_weight * params["max_soft_blend"], 0.0, params["max_soft_blend"])
    levels = (
        pd.to_numeric(out.get("channel_level", 0), errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    y_channel = np.array(
        [channel_targets.get(int(level), y) for level, y in zip(levels, y_brt)],
        dtype=float,
    )
    y_target = np.maximum(y_brt, y_channel)
    y_final = (1.0 - blend_weight) * y_brt + blend_weight * y_target
    out["channel_target_thickness"] = y_channel
    out["channel_blend_weight"] = blend_weight
    out[out_col] = y_final
    out["delta_channel_minus_raw"] = out[out_col] - y_brt
    return out


def to_grid_array(df: pd.DataFrame, col: str, *, params) -> np.ndarray:
    g = np.full((params["grid_nrow"], params["grid_ncol"]), np.nan)
    r = pd.to_numeric(df["grid_row"], errors="coerce")
    c = pd.to_numeric(df["grid_col"], errors="coerce")
    v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    ok = (r.notna() & c.notna()).to_numpy()
    ri = r.to_numpy(dtype=float)[ok].astype(int)
    ci = c.to_numpy(dtype=float)[ok].astype(int)
    g[ri, ci] = v[ok]
    return g


def _shift_bool(mask: np.ndarray, di: int, dj: int) -> np.ndarray:
    s = np.roll(np.roll(mask, di, axis=0), dj, axis=1)
    if di > 0:
        s[:di, :] = False
    elif di < 0:
        s[di:, :] = False
    if dj > 0:
        s[:, :dj] = False
    elif dj < 0:
        s[:, dj:] = False
    return s


def targeted_bridge(seed: np.ndarray, domain: np.ndarray, rounds: int) -> np.ndarray:
    m = seed.copy()
    out = np.zeros_like(seed)
    for _ in range(max(int(rounds), 0)):
        bridge = np.zeros_like(m)
        for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
            bridge |= _shift_bool(m, di, dj) & _shift_bool(m, -di, -dj)
        new = domain & ~m & bridge
        if not new.any():
            break
        out |= new
        m = m | new
    return out


def morph_band_floor(dist: np.ndarray, floors: np.ndarray, *, params) -> np.ndarray:
    d = np.asarray(dist, dtype=float)
    out = np.zeros(d.shape, dtype=float)
    flat, oflat = (d.ravel(), out.ravel())
    for i in range(len(floors)):
        m = (flat > params["morph_band_edges"][i]) & (flat <= params["morph_band_edges"][i + 1])
        oflat[m] = floors[i]
    return out


def build_morph_floors(
    train_dist: np.ndarray, y_train: np.ndarray, *, params
) -> np.ndarray:
    d = np.asarray(train_dist, dtype=float)
    y = np.asarray(y_train, dtype=float)
    floors = []
    for i, _ in enumerate(params["morph_band_names"]):
        m = (
            np.isfinite(d)
            & (d > params["morph_band_edges"][i])
            & (d <= params["morph_band_edges"][i + 1])
        )
        v = y[m]
        v = v[np.isfinite(v)]
        last = i == len(params["morph_band_names"]) - 1
        floors.append(0.0 if last or v.size == 0 else float(np.percentile(v, 25)))
    return np.array(floors, dtype=float)


def morph_finish_grid(
    c2_grid: np.ndarray,
    dist_grid: np.ndarray,
    floors: np.ndarray,
    cut: float,
    band_max_d: float,
    rounds: int,
    *,
    params,
):
    c2 = np.asarray(c2_grid, dtype=float)
    dist = np.asarray(dist_grid, dtype=float)
    band = np.isfinite(dist) & (dist <= band_max_d)
    floor_g = morph_band_floor(dist, floors, params=params)
    lifted = c2 + params["morph_floor_blend"] * (floor_g - c2)
    s_floor = np.where(np.isfinite(c2), np.maximum(c2, lifted), c2)
    seed = np.isfinite(s_floor) & (s_floor >= cut)
    domain = band & np.isfinite(s_floor)
    mask_bridge = targeted_bridge(seed, domain, rounds=rounds)
    final = np.where(mask_bridge, np.maximum(s_floor, cut * 1.002), s_floor)
    mask_floor = np.isfinite(c2) & (s_floor - c2 > 1e-09)
    return (final, mask_bridge, mask_floor)


def morph_finish_points(
    base: np.ndarray,
    dist_vals: np.ndarray,
    floors: np.ndarray,
    cut: float,
    mask_bridge: np.ndarray | None = None,
    rows: np.ndarray | None = None,
    cols: np.ndarray | None = None,
    *,
    params,
) -> np.ndarray:
    b = np.asarray(base, dtype=float)
    floor_pt = morph_band_floor(np.asarray(dist_vals, dtype=float), floors, params=params)
    out = np.maximum(b, b + params["morph_floor_blend"] * (floor_pt - b))
    if mask_bridge is not None and rows is not None and (cols is not None):
        r = np.asarray(rows, dtype=int)
        c = np.asarray(cols, dtype=int)
        ok = (
            (r >= 0)
            & (r < mask_bridge.shape[0])
            & (c >= 0)
            & (c < mask_bridge.shape[1])
        )
        fill = np.zeros(out.shape, dtype=bool)
        fill[ok] = mask_bridge[r[ok], c[ok]]
        out = np.where(fill, np.maximum(out, cut * 1.002), out)
    return out


def morph_structure_metrics(
    grid: np.ndarray, cut: float, *, params
) -> dict[str, float]:
    g = np.asarray(grid, dtype=float)
    b = np.isfinite(g) & (g >= cut)
    lab = cc_label(b, connectivity=2)
    n = int(lab.max())
    if b.sum():
        sizes = np.bincount(lab.ravel())[1:]
        share = float(sizes.max() / sizes.sum())
        sk = skeletonize(b)
        nb = sum(
            (
                _shift_bool(sk, i, j)
                for i in (-1, 0, 1)
                for j in (-1, 0, 1)
                if (i, j) != (0, 0)
            )
        )
        ends = int((sk & (nb == 1)).sum())
    else:
        share, ends = (0.0, 0)
    return {
        "sand_area_km2": float(b.sum() * params["cell_km2"]),
        "connected_components": n,
        "largest_component_fraction": share,
        "skeleton_endpoints": ends,
    }


def train_brt(train_df: pd.DataFrame, *, params):
    x_train, _ = make_arrays(train_df, train_df.head(1), params=params)
    y_train = pd.to_numeric(train_df[params["target_col"]], errors="coerce").to_numpy(dtype=float)
    model = GradientBoostingRegressor(**params["brt_params"])
    model.fit(x_train, y_train, sample_weight=sample_weight(train_df))
    return model


def predict_brt(
    model, train_df: pd.DataFrame, pred_df: pd.DataFrame, *, params
) -> np.ndarray:
    _, x_pred = make_arrays(train_df, pred_df, params=params)
    return np.clip(model.predict(x_pred), 0.0, None)


def evaluate_validation(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    bandwidth: dict[int, float],
    channel_data: ChannelData,
    *,
    params,
) -> dict[str, float]:
    model = train_brt(train_df, params=params)
    valid_pred = valid_df.copy()
    valid_pred["pred_sand_thickness_raw"] = predict_brt(
        model, train_df, valid_pred, params=params
    )
    valid_pred = calculate_channel_distance(valid_pred, channel_data, params=params)
    valid_pred = calculate_channel_weights_for_points(
        valid_pred, bandwidth, decay_mode=params["decay_mode"], params=params
    )
    targets = build_channel_targets(
        pd.to_numeric(train_df[params["target_col"]], errors="coerce").to_numpy(dtype=float),
        min_samples=params["min_target_samples"],
        params=params,
    )
    valid_pred = apply_channel_soft_constraint(
        valid_pred, "pred_sand_thickness_raw", targets, params=params
    )
    y_true = pd.to_numeric(valid_pred[params["target_col"]], errors="coerce").to_numpy(
        dtype=float
    )
    raw_metrics = calc_metrics(y_true, valid_pred["pred_sand_thickness_raw"])
    cc_metrics = calc_metrics(y_true, valid_pred["pred_sand_thickness_channel_constrained"])
    return {
        "raw_R2": raw_metrics["R2"],
        "raw_RMSE": raw_metrics["RMSE"],
        "raw_MAE": raw_metrics["MAE"],
        "channel_R2": cc_metrics["R2"],
        "channel_RMSE": cc_metrics["RMSE"],
        "channel_MAE": cc_metrics["MAE"],
    }


def search_best_channel_bandwidth(
    train_df: pd.DataFrame, valid_df: pd.DataFrame, channel_data: ChannelData, *, params
) -> tuple[dict[int, float], pd.DataFrame]:
    rows = []
    for b1 in params["bandwidth_search_space"][1]:
        for b2 in params["bandwidth_search_space"][2]:
            for b3 in params["bandwidth_search_space"][3]:
                bandwidth = {1: b1, 2: b2, 3: b3}
                if not b1 >= b2 >= b3:
                    continue
                met = evaluate_validation(
                    train_df, valid_df, bandwidth, channel_data, params=params
                )
                rows.append({"B1": b1, "B2": b2, "B3": b3, **met})
    result = pd.DataFrame(rows).sort_values(["channel_RMSE", "channel_MAE"], ascending=True)
    best = result.iloc[0]
    return ({1: float(best["B1"]), 2: float(best["B2"]), 3: float(best["B3"])}, result)


def run_channel_model(
    train_all,
    valid_df,
    predict_df,
    ch_df,
    grid_df,
    channel_data,
    train_labels,
    *,
    params,
):
    tables = {}
    grid_df = grid_df.rename(columns={v: k for k, v in params["grid_attr_map"].items()})
    train_all = attach_channel_feature(
        train_all, ch_df, grid_df, "training samples", params=params
    )
    valid_df = attach_channel_feature(
        valid_df, ch_df, grid_df, "validation samples", params=params
    )
    predict_df = attach_channel_feature(
        predict_df, ch_df, grid_df, "prediction grid", params=params
    )
    val_wells = sorted(valid_df["source_well"].astype(str).unique())
    train_all = train_all[
        ~train_all["source_well"].astype(str).isin(val_wells)
    ].reset_index(drop=True)
    validate_feature_columns(train_all, "training samples", params=params)
    validate_feature_columns(valid_df, "validation samples", params=params)
    bandwidth = params["channel_bandwidth"].copy()
    validate_bandwidths(bandwidth)
    gss = GroupShuffleSplit(
        n_splits=1, test_size=params["test_size"], random_state=params["random_state"]
    )
    tr_idx, te_idx = next(gss.split(train_all, groups=train_all["source_well"].astype(str)))
    train_df = train_all.iloc[tr_idx].reset_index(drop=True)
    test_df = train_all.iloc[te_idx].reset_index(drop=True)
    if params["enable_bandwidth_search"] and channel_data.valid:
        bandwidth, search_df = search_best_channel_bandwidth(
            train_df, test_df, channel_data, params=params
        )
        tables["00_bandwidth_search.csv"] = search_df
    model = train_brt(train_df, params=params)
    y_train = pd.to_numeric(train_df[params["target_col"]], errors="coerce").to_numpy(dtype=float)
    train_ch = calculate_channel_weights_for_points(
        calculate_channel_distance(train_df.copy(), channel_data, params=params),
        bandwidth,
        decay_mode=params["decay_mode"],
        params=params,
    )
    train_levels = (
        pd.to_numeric(train_ch["channel_level"], errors="coerce")
        .fillna(0)
        .to_numpy(int)
    )
    channel_targets = build_channel_targets(
        y_train,
        levels=train_levels,
        min_samples=params["min_target_samples"],
        params=params,
    )
    outputs = {}
    for name, df in [
        ("train", train_df),
        ("test", test_df),
        ("external_validation", valid_df),
    ]:
        out = df.copy()
        out["pred_sand_thickness_raw"] = predict_brt(model, train_df, out, params=params)
        out = calculate_channel_distance(out, channel_data, params=params)
        out = calculate_channel_weights_for_points(
            out, bandwidth, decay_mode=params["decay_mode"], params=params
        )
        out = apply_channel_soft_constraint(
            out, "pred_sand_thickness_raw", channel_targets, params=params
        )
        outputs[name] = out
    model_full = train_brt(train_all, params=params)
    y_all = pd.to_numeric(train_all[params["target_col"]], errors="coerce").to_numpy(dtype=float)
    train_all_ch = calculate_channel_weights_for_points(
        calculate_channel_distance(train_all.copy(), channel_data, params=params),
        bandwidth,
        decay_mode=params["decay_mode"],
        params=params,
    )
    levels_full = (
        pd.to_numeric(train_all_ch["channel_level"], errors="coerce")
        .fillna(0)
        .to_numpy(int)
    )
    targets_full = build_channel_targets(
        y_all,
        levels=levels_full,
        min_samples=params["min_target_samples"],
        params=params,
    )
    full_pred = predict_df.copy()
    if all((c in full_pred.columns for c in params["feature_cols"])):
        full_pred["pred_sand_thickness_raw"] = predict_brt(
            model_full, train_all, full_pred, params=params
        )
    elif "pred_sand_thickness_raw" not in full_pred.columns:
        raise ValueError(
            "Prediction input lacks the five required features and pred_sand_thickness_raw."
        )
    else:
        missing = [c for c in params["feature_cols"] if c not in full_pred.columns]
    full_pred = calculate_channel_distance(full_pred, channel_data, params=params)
    full_pred = calculate_channel_weights_for_points(
        full_pred, bandwidth, decay_mode=params["decay_mode"], params=params
    )
    full_pred = apply_channel_soft_constraint(
        full_pred, "pred_sand_thickness_raw", targets_full, params=params
    )
    C2_COL = "pred_sand_thickness_channel_constrained"
    mask_bridge = None
    morph_floors = np.zeros(len(params["morph_band_names"]), dtype=float)
    morph_cut = float("nan")
    if params["enable_morph_finish"]:
        morph_floors = build_morph_floors(
            train_ch["distance_to_channel"].to_numpy(dtype=float),
            y_train,
            params=params,
        )
        if params["morph_cut"] == "auto":
            lab_y = pd.to_numeric(
                train_labels[params["target_col"]], errors="coerce"
            ).to_numpy(dtype=float)
            lab_y = lab_y[np.isfinite(lab_y)]
            morph_cut = float(np.quantile(lab_y, params["morph_cut_quantile"]))
        else:
            morph_cut = float(params["morph_cut"])
        c2_grid = to_grid_array(full_pred, C2_COL, params=params)
        dist_grid = to_grid_array(full_pred, "distance_to_channel", params=params)
        raw_grid = to_grid_array(full_pred, "pred_sand_thickness_raw", params=params)
        final_grid, mask_bridge, mask_floor = morph_finish_grid(
            c2_grid,
            dist_grid,
            morph_floors,
            morph_cut,
            params["morph_band_max_d"],
            params["morph_bridge_rounds"],
            params=params,
        )
        rr = full_pred["grid_row"].to_numpy(int)
        cc = full_pred["grid_col"].to_numpy(int)
        full_pred[params["morph_col_final"]] = final_grid[rr, cc]
        full_pred[params["morph_col_delta"]] = pd.to_numeric(
            full_pred[params["morph_col_final"]], errors="coerce"
        ) - pd.to_numeric(full_pred[C2_COL], errors="coerce")
        full_pred[params["morph_col_delta_total"]] = pd.to_numeric(
            full_pred[params["morph_col_final"]], errors="coerce"
        ) - pd.to_numeric(full_pred["pred_sand_thickness_raw"], errors="coerce")
        for name, out in outputs.items():
            fin = morph_finish_points(
                out[C2_COL].to_numpy(dtype=float),
                out["distance_to_channel"].to_numpy(dtype=float),
                morph_floors,
                morph_cut,
                mask_bridge,
                out["grid_row"].to_numpy(int),
                out["grid_col"].to_numpy(int),
                params=params,
            )
            out[params["morph_col_final"]] = fin
            out[params["morph_col_delta"]] = fin - pd.to_numeric(
                out[C2_COL], errors="coerce"
            ).to_numpy(dtype=float)
            out[params["morph_col_delta_total"]] = fin - pd.to_numeric(
                out["pred_sand_thickness_raw"], errors="coerce"
            ).to_numpy(dtype=float)
        struct_df = pd.DataFrame(
            [
                {
                    "scheme": nm,
                    "threshold_m": round(morph_cut, 4),
                    **morph_structure_metrics(grid, morph_cut, params=params),
                }
                for nm, grid in (
                    ("raw", raw_grid),
                    ("c2_baseline", c2_grid),
                    ("final_morphology", final_grid),
                )
            ]
        )
        tables["17_morphology_metrics.csv"] = struct_df
    metrics_rows = []
    for name, out in outputs.items():
        y_true = pd.to_numeric(out[params["target_col"]], errors="coerce").to_numpy(dtype=float)
        variants = [
            ("BRT_raw", "pred_sand_thickness_raw"),
            ("BRT_channel_constrained", C2_COL),
        ]
        if params["enable_morph_finish"]:
            variants.append(("BRT_morph_finished", params["morph_col_final"]))
        for label, col in variants:
            metrics_rows.append(
                {"dataset": name, "prediction": label, **calc_metrics(y_true, out[col])}
            )
    metrics_df = pd.DataFrame(metrics_rows)
    tables["01_model_metrics.csv"] = metrics_df
    tables["02_training_predictions.csv"] = outputs["train"]
    tables["03_test_predictions.csv"] = outputs["test"]
    tables["04_validation_predictions.csv"] = outputs["external_validation"]
    tables["05_grid_predictions.csv"] = full_pred
    tables["12_constraint_parameters.csv"] = pd.DataFrame(
        {
            "channel_level": [1, 2, 3],
            "bandwidth_m": [bandwidth[level] for level in (1, 2, 3)],
            "target_thickness": [targets_full[level] for level in (1, 2, 3)],
        }
    )
    return (tables, outputs, full_pred)


def run(*, params) -> None:
    params["out_dir"].mkdir(parents=True, exist_ok=True)
    train_all = (
        read_csv(params["train_csv"], options=params["csv_options"])
        .dropna(subset=[params["target_col"]])
        .reset_index(drop=True)
    )
    valid_df = (
        read_csv(params["valid_csv"], options=params["csv_options"])
        .dropna(subset=[params["target_col"]])
        .reset_index(drop=True)
    )
    predict_df = read_csv(params["predict_csv"], options=params["csv_options"])
    ch_df = read_csv(params["channel_grid_csv"], options=params["csv_options"])
    grid_df = read_csv(
        params["grid_attr_csv"],
        options=params["csv_options"],
    )[params["key_cols"] + list(params["grid_attr_map"].values())]
    channel_data = load_channel_data(params["gis_database"], params["gis_layer"])
    tables, outputs, full_pred = run_channel_model(
        train_all,
        valid_df,
        predict_df,
        ch_df,
        grid_df,
        channel_data,
        train_all.copy(),
        params=params,
    )
    for filename, table in tables.items():
        write_csv(table, params["out_dir"] / filename)
    C2_COL = "pred_sand_thickness_channel_constrained"
    if params["enable_morph_finish"]:
        plot_morphology(full_pred, C2_COL, params=params)
    plot_scatter(
        outputs["test"],
        params["target_col"],
        "pred_sand_thickness_raw",
        params["out_dir"] / "06_test_raw_scatter.png",
        "test set BRT raw",
    )
    plot_scatter(
        outputs["test"],
        params["target_col"],
        "pred_sand_thickness_channel_constrained",
        params["out_dir"] / "07_test_constrained_scatter.png",
        "test set channel constraintafter",
    )
    plot_scatter(
        outputs["external_validation"],
        params["target_col"],
        "pred_sand_thickness_channel_constrained",
        params["out_dir"] / "08_validation_constrained_scatter.png",
        "external validation channel constraintafter",
    )
