from metrics import regression_metrics as calc_metrics
from plotting import plot_ablation
from dataset import read_csv
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler


def prepare_data(
    real, virtual, ch, *, params
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cn_map = {
        "sand_thickness_cn": "sand_thickness",
        "total_energy": "sum_of_energy_mean",
        "average_energy": "average_energy_mean",
        "minimum_amplitude": "minimum_amplitude_mean",
        "rms_amplitude": "rms_mean",
        "std_amplitude": "standard_deviation_of_amplitude_mean",
    }
    virtual.rename(
        columns={k: v for k, v in cn_map.items() if k in virtual.columns}, inplace=True
    )
    ch_cols = [
        c
        for c in ch.columns
        if c
        in (
            "grid_row",
            "grid_col",
            "channel_weight_2000",
            "channel_zone",
            "in_channel_3000",
            "distance_to_channel",
        )
    ]
    ch = ch[ch_cols].copy()
    for c in ch_cols:
        if c not in ("grid_row", "grid_col"):
            ch[c] = pd.to_numeric(ch[c], errors="coerce").fillna(0.0)
    ch["grid_row"] = pd.to_numeric(ch["grid_row"], errors="coerce")
    ch["grid_col"] = pd.to_numeric(ch["grid_col"], errors="coerce")
    for df in [real, virtual]:
        df["grid_row"] = pd.to_numeric(df["grid_row"], errors="coerce")
        df["grid_col"] = pd.to_numeric(df["grid_col"], errors="coerce")
    real = real.merge(ch, on=["grid_row", "grid_col"], how="left")
    virtual = virtual.merge(ch, on=["grid_row", "grid_col"], how="left")
    for df in [real, virtual]:
        for c in ["channel_weight_2000", "channel_zone", "in_channel_3000"]:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    real[params["sample_type_col"]] = "real"
    virtual[params["sample_type_col"]] = "virtual"
    real[params["source_well_col"]] = (real[params["source_well_col"]].astype(str).str.strip())
    virtual[params["source_well_col"]] = (
        virtual[params["source_well_col"]].astype(str).str.strip()
    )
    virtual["neighbor_order"] = (
        pd.to_numeric(virtual.get("neighbor_order", 1), errors="coerce")
        .fillna(1)
        .astype(int)
    )
    for df in [real, virtual]:
        df[params["target_col"]] = pd.to_numeric(df[params["target_col"]], errors="coerce")
        for c in params["feature_cols"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=[params["target_col"]] + params["feature_cols"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return (real, virtual, ch)


def compute_geo_weight(
    cand_cw: float,
    cand_in: float,
    cand_zone: float,
    well_cw: np.ndarray,
    well_in: np.ndarray,
    well_zone: np.ndarray,
) -> np.ndarray:
    sigma_geo = 0.35
    strength_sim = np.exp(-(((cand_cw - well_cw) / sigma_geo) ** 2))
    in_sim = np.where(well_in == cand_in, 1.0, 0.7)
    zone_sim = np.where(np.abs(well_zone - cand_zone) < 1e-06, 1.0, 0.75)
    return strength_sim**0.5 * in_sim**0.3 * zone_sim**0.2


def compute_virtual_labels_and_weights(
    real: pd.DataFrame, virtual: pd.DataFrame, use_channel: bool, *, params
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    real_feat = scaler.fit_transform(real[params["feature_cols"]].to_numpy(dtype=float))
    virtual_feat = scaler.transform(virtual[params["feature_cols"]].to_numpy(dtype=float))
    real_xy = real[["X_center", "Y_center"]].to_numpy(dtype=float)
    virtual_xy = virtual[["X_center", "Y_center"]].to_numpy(dtype=float)
    real_y = real[params["target_col"]].to_numpy(dtype=float)
    order = virtual["neighbor_order"].to_numpy()
    base_weight = np.array([{1: 0.25, 2: 0.12, 3: 0.08}.get(int(o), 0.08) for o in order])
    brt_label = GradientBoostingRegressor(
        n_estimators=800,
        learning_rate=0.01,
        max_depth=3,
        min_samples_leaf=2,
        min_samples_split=10,
        subsample=0.7,
        max_features=0.6,
        loss="squared_error",
        random_state=42,
    )
    brt_label.fit(real[params["feature_cols"]].fillna(0), real_y)
    pred_labels = brt_label.predict(virtual[params["feature_cols"]].fillna(0))
    real_by_well = real.set_index(params["source_well_col"], drop=False)
    src_real = real_by_well.loc[virtual[params["source_well_col"]]].reset_index(drop=True)
    src_vals = src_real[params["target_col"]].to_numpy(dtype=float)
    labels = np.zeros(len(virtual), dtype=float)
    for i in range(len(virtual)):
        attr_d = np.linalg.norm(real_feat - virtual_feat[i], axis=1)
        space_d = np.linalg.norm(real_xy - virtual_xy[i], axis=1)
        sigma_a = max(float(np.nanpercentile(attr_d, 75)), 1e-06)
        w_a = np.exp(-(attr_d**2) / (2.0 * sigma_a**2))
        w_s = np.exp(-(space_d**2) / (2.0 * params["sigma_space"] ** 2))
        alpha = (0.3 * np.exp(-space_d[0] ** 2 / (2.0 * params["sigma_space"] ** 2)) * w_a[0])
        labels[i] = (1 - min(alpha, 0.3)) * pred_labels[i] + min(alpha, 0.3) * src_vals[
            i
        ]
    labels = np.clip(labels, 0.0, None)
    real_cw = real["channel_weight_2000"].to_numpy(dtype=float)
    real_in = (real["in_channel_3000"].to_numpy(dtype=float) > 0).astype(float)
    real_zone = real["channel_zone"].to_numpy(dtype=float)
    train_weights = np.zeros(len(virtual), dtype=float)
    for i in range(len(virtual)):
        attr_dist = np.linalg.norm(real_feat - virtual_feat[i], axis=1)
        space_dist = np.linalg.norm(real_xy - virtual_xy[i], axis=1)
        nearest = np.argsort(attr_dist + space_dist / params["sigma_space"])[
            : params["k_neighbor"]
        ]
        sigma_a = max(float(np.nanpercentile(attr_dist[nearest], 75)), 1e-06)
        w_a = np.exp(-attr_dist[nearest] ** 2 / (2.0 * sigma_a**2))
        w_s = np.exp(-space_dist[nearest] ** 2 / (2.0 * params["sigma_space"] ** 2))
        if use_channel:
            cand_cw = float(virtual.iloc[i].get("channel_weight_2000", 0.0))
            cand_in = float(virtual.iloc[i].get("in_channel_3000", 0.0)) > 0
            cand_zone = float(virtual.iloc[i].get("channel_zone", 0.0))
            w_g = compute_geo_weight(
                cand_cw,
                cand_in,
                cand_zone,
                real_cw[nearest],
                real_in[nearest],
                real_zone[nearest],
            )
            joint_base = w_a**0.6 * w_s**0.4
            joint = joint_base * w_g
            s_a = float(np.average(w_a, weights=joint)) if joint.sum() > 0 else 0
            s_s = float(np.average(w_s, weights=joint)) if joint.sum() > 0 else 0
            s_g = float(np.average(w_g, weights=joint)) if joint.sum() > 0 else 0
            confidence = 0.4 * s_a + 0.3 * s_s + 0.3 * s_g
        else:
            joint = w_a**0.6 * w_s**0.4
            s_a = float(np.average(w_a, weights=joint)) if joint.sum() > 0 else 0
            s_s = float(np.average(w_s, weights=joint)) if joint.sum() > 0 else 0
            confidence = 0.6 * s_a + 0.4 * s_s
        train_weights[i] = np.clip(base_weight[i] * confidence, 0.0, 1.0)
    return (labels, train_weights)


def apply_channel_soft_constraint(
    y_pred: np.ndarray, df: pd.DataFrame, y_train: np.ndarray
) -> np.ndarray:
    if "channel_weight_2000" not in df.columns:
        return y_pred
    cw = (
        pd.to_numeric(df["channel_weight_2000"], errors="coerce")
        .fillna(0.0)
        .clip(0, 1)
        .to_numpy(dtype=float)
    )
    cz = (
        pd.to_numeric(df.get("channel_zone", 0), errors="coerce")
        .fillna(0)
        .to_numpy(dtype=int)
    )
    q_high = float(np.nanquantile(y_train, 0.8))
    q_mid = float(np.nanquantile(y_train, 0.6))
    q_low = float(np.nanquantile(y_train, 0.4))
    y_brt = y_pred.copy()
    y_ch = np.where(cz == 1, q_high, np.where(cz == 2, q_mid, q_low))
    blend = np.clip(cw * 0.6, 0.0, 0.75)
    y_target = np.maximum(y_brt, y_ch)
    return (1.0 - blend) * y_brt + blend * y_target


def run_ablation(real, virtual, ch_grid, *, params):
    labels_noch, w_noch = compute_virtual_labels_and_weights(
        real, virtual, use_channel=False, params=params
    )
    labels_ch, w_ch = compute_virtual_labels_and_weights(
        real, virtual, use_channel=True, params=params
    )
    base_cols = [
        params["source_well_col"],
        "grid_row",
        "grid_col",
        "X_center",
        "Y_center",
        params["target_col"],
        params["sample_type_col"],
        "neighbor_order",
        *params["feature_cols"],
    ]
    wells = real[params["source_well_col"]].astype(str).to_numpy()
    rkf = RepeatedKFold(
        n_splits=params["n_splits"],
        n_repeats=params["n_repeats"],
        random_state=params["random_state"],
    )
    schemes = [
        ("C0 No channel prior", False, False),
        ("C1 Sampling prior", True, False),
        ("C2 Prediction constraint", False, True),
        ("C3 Both stages", True, True),
    ]
    all_pred = []
    for scheme_name, use_ch, use_pp in schemes:
        chosen_labels = labels_ch if use_ch else labels_noch
        chosen_weights = w_ch if use_ch else w_noch
        for fi, (train_idx, test_idx) in enumerate(rkf.split(wells), 1):
            rep = (fi - 1) // params["n_splits"] + 1
            fold = (fi - 1) % params["n_splits"] + 1
            train_wells_set = set(wells[train_idx])
            test_wells_set = set(wells[test_idx])
            train_real = real[
                real[params["source_well_col"]].astype(str).isin(train_wells_set)
            ].copy()
            train_real[params["target_col"]] = pd.to_numeric(
                train_real[params["target_col"]], errors="coerce"
            )
            train_real["neighbor_order"] = 0
            is_src = (virtual[params["source_well_col"]].astype(str).isin(train_wells_set))
            is_pos = chosen_weights > params["min_positive_weight"]
            train_virt = virtual[is_src & is_pos].copy()
            train_virt[params["target_col"]] = chosen_labels[is_src & is_pos]
            train_virt["_weight"] = chosen_weights[is_src & is_pos]
            train_df = pd.concat(
                [train_real[base_cols], train_virt[base_cols + ["_weight"]]],
                ignore_index=True,
            )
            test_df = real[
                real[params["source_well_col"]].astype(str).isin(test_wells_set)
            ].copy()
            test_df["neighbor_order"] = 0
            x_train = (
                train_df[params["feature_cols"]]
                .fillna(train_df[params["feature_cols"]].median())
                .fillna(0.0)
            )
            x_test = (
                test_df[params["feature_cols"]]
                .fillna(train_df[params["feature_cols"]].median())
                .fillna(0.0)
            )
            y_train = train_df[params["target_col"]].to_numpy(dtype=float)
            y_test = test_df[params["target_col"]].to_numpy(dtype=float)
            sw = np.ones(len(train_df), dtype=float)
            if "_weight" in train_df.columns:
                is_v = train_df[params["sample_type_col"]].eq("virtual").to_numpy()
                sw[is_v] = train_df.loc[is_v, "_weight"].to_numpy(dtype=float)
            model_params = dict(params["brt_params"])
            model_params["random_state"] = params["random_state"] + rep * 100 + fold
            model = GradientBoostingRegressor(**model_params)
            model.fit(x_train, y_train, sample_weight=sw.clip(0))
            pred = model.predict(x_test)
            if use_pp:
                test_merged = test_df.merge(
                    ch_grid[
                        ["grid_row", "grid_col", "channel_weight_2000", "channel_zone"]
                    ],
                    on=["grid_row", "grid_col"],
                    how="left",
                )
                pred = apply_channel_soft_constraint(pred, test_merged, y_train)
            all_pred.append(
                pd.DataFrame(
                    {
                        "experiment": scheme_name,
                        "repeat": rep,
                        "y_true": y_test,
                        "y_pred": pred,
                    }
                )
            )
    predictions = pd.concat(all_pred, ignore_index=True)
    repeat_rows = []
    for (exp, rep), part in predictions.groupby(["experiment", "repeat"], sort=False):
        m = calc_metrics(
            part["y_true"].to_numpy(dtype=float), part["y_pred"].to_numpy(dtype=float)
        )
        repeat_rows.append({"experiment": exp, "repeat": rep, **m})
    repeat_metrics = pd.DataFrame(repeat_rows)
    summary = repeat_metrics.groupby("experiment", as_index=False).agg(
        R2=("R2", "mean"),
        R2_std=("R2", "std"),
        RMSE=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        MAE=("MAE", "mean"),
        MAE_std=("MAE", "std"),
    )
    return (predictions, repeat_metrics, summary)


def run(*, params):
    params["out_dir"].mkdir(parents=True, exist_ok=True)
    real = read_csv(params["real_csv"], options=params["csv_options"])
    virtual = read_csv(params["virtual_csv"], options=params["csv_options"])
    channels = read_csv(params["channel_grid_csv"], options=params["csv_options"])
    real, virtual, channels = prepare_data(real, virtual, channels, params=params)
    predictions, repeat_metrics, summary = run_ablation(real, virtual, channels, params=params)
    summary.to_csv(params["out_dir"] / "table5_2_summary.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(
        params["out_dir"] / "table5_2_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_ablation(predictions, repeat_metrics, summary, params["out_dir"])
