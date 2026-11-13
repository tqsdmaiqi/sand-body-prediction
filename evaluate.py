from plotting import plot_evaluation
from pathlib import Path
from dataset import read_csv
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def find_file(name: str, *, params) -> Path:
    return params["channel_output"] / name


def metrics(y, p) -> dict[str, float]:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return {
        "R2": float(r2_score(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": float(mean_absolute_error(y, p)),
        "Bias": float(np.mean(p - y)),
    }


def bootstrap_ci(diff: np.ndarray, *, n_boot, params) -> tuple[float, float]:
    diff = np.asarray(diff, float)
    if len(diff) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(params["random_seed"])
    means = rng.choice(diff, (n_boot, len(diff)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, params["confidence_percentiles"])
    return (float(lo), float(hi))


def paired(y, p_raw, p_fin, groups=None, *, params) -> dict:
    e_raw = np.abs(np.asarray(p_raw, float) - np.asarray(y, float))
    e_fin = np.abs(np.asarray(p_fin, float) - np.asarray(y, float))
    diff = e_raw - e_fin
    out = {
        "sample_mae_improvement": float(diff.mean()),
        "samples_improved": int((diff > 1e-09).sum()),
        "samples_worsened": int((diff < -1e-09).sum()),
        "samples_unchanged": int((np.abs(diff) <= 1e-09).sum()),
        "sample_count": int(len(diff)),
    }
    if len(diff) >= 2 and diff.std() > 0:
        out["sample_paired_t_p"] = float(stats.ttest_rel(e_raw, e_fin)[1])
        lo, hi = bootstrap_ci(diff, n_boot=params["n_bootstrap"], params=params)
        out["sample_ci_lower"], out["sample_ci_upper"] = (lo, hi)
    if groups is not None:
        g = pd.DataFrame({"w": np.asarray(groups), "d": diff}).groupby("w")["d"].mean()
        gd = g.to_numpy(float)
        out["well_count"] = int(len(gd))
        out["well_mean_mae_improvement"] = float(gd.mean()) if len(gd) else np.nan
        out["wells_improved"] = int((gd > 1e-09).sum())
        out["wells_worsened"] = int((gd < -1e-09).sum())
        out["wells_unchanged"] = int((np.abs(gd) <= 1e-09).sum())
        if len(gd) >= 2 and gd.std() > 0:
            out["well_paired_t_p"] = float(stats.ttest_1samp(gd, 0.0)[1])
            lo, hi = bootstrap_ci(gd, n_boot=params["n_bootstrap"], params=params)
            out["well_ci_lower"], out["well_ci_upper"] = (lo, hi)
    return out


def run(*, params) -> None:
    params["out_dir"].mkdir(parents=True, exist_ok=True)
    ch = read_csv(
        find_file("05_grid_predictions.csv", params=params),
        options=params["csv_options"],
    )[params["key"] + ["channel_zone"]]
    data: dict[str, pd.DataFrame] = {}
    for name, fn in params["datasets"]:
        d = read_csv(find_file(fn, params=params), options=params["csv_options"])
        if "channel_zone" not in d.columns:
            d = d.drop(columns=["channel_zone"], errors="ignore").merge(
                ch, on=params["key"], how="left"
            )
        d = d.dropna(
            subset=[params["y_col"], params["col_raw"], params["col_fin"]]
        ).reset_index(drop=True)
        data[name] = d
    full = read_csv(
        find_file("05_grid_predictions.csv", params=params),
        options=params["csv_options"],
    )
    fr = pd.to_numeric(full[params["col_raw"]], errors="coerce").to_numpy(float)
    ff = pd.to_numeric(full[params["col_fin"]], errors="coerce").to_numpy(float)
    fd = ff - fr
    fz = pd.to_numeric(full["channel_zone"], errors="coerce").fillna(-1).to_numpy(int)
    fl = pd.to_numeric(full["channel_level"], errors="coerce").fillna(0).to_numpy(int)
    nz = int((np.abs(fd) > 1e-09).sum())
    rows = []
    for z in (0, 1, 2):
        m = fz == z
        if not m.any():
            continue
        rows.append(
            {
                "zone": params["zone_name"][z],
                "cell_count": int(m.sum()),
                "changed_cell_count": int((np.abs(fd[m]) > 1e-09).sum()),
                "nonzero_fraction": float((np.abs(fd[m]) > 1e-09).mean()),
                "delta_mean_m": float(np.nanmean(fd[m])),
                "delta_max_m": float(np.nanmax(fd[m])),
            }
        )
    zdf = pd.DataFrame(rows)
    lrows = []
    for lv in (0, 1, 2, 3):
        m = fl == lv
        if m.any():
            lrows.append(
                {
                    "channel_level": lv,
                    "cell_count": int(m.sum()),
                    "changed_cell_fraction": float((np.abs(fd[m]) > 1e-09).mean()),
                    "delta_mean_m": float(np.nanmean(fd[m])),
                }
            )
    ldf = pd.DataFrame(lrows)
    mrows = []
    for name, d in data.items():
        y = d[params["y_col"]].to_numpy(float)
        pr = d[params["col_raw"]].to_numpy(float)
        pf = d[params["col_fin"]].to_numpy(float)
        grp = d["source_well"].astype(str).to_numpy() if "source_well" in d else None
        a, b = (metrics(y, pr), metrics(y, pf))
        pv = paired(y, pr, pf, grp, params=params)
        mrows.append(
            {
                "dataset": name,
                "n": len(d),
                **{f"raw_{k}": v for k, v in a.items()},
                **{f"constrained_{k}": v for k, v in b.items()},
                "RMSE_change": b["RMSE"] - a["RMSE"],
                "MAE_change": b["MAE"] - a["MAE"],
                "bias_change": b["Bias"] - a["Bias"],
                **pv,
            }
        )
    mdf = pd.DataFrame(mrows)
    mdf.to_csv(params["out_dir"] / "19_constraint_metrics.csv", index=False, encoding="utf-8-sig")
    prows = []
    for name, d in data.items():
        y = d[params["y_col"]].to_numpy(float)
        pr = d[params["col_raw"]].to_numpy(float)
        pf = d[params["col_fin"]].to_numpy(float)
        dt = d[params["col_delta"]].to_numpy(float)
        z = (
            pd.to_numeric(d["channel_zone"], errors="coerce").fillna(-1).to_numpy(int)
            if "channel_zone" in d
            else np.full(len(d), -1)
        )
        base = metrics(y, pr)
        real = metrics(y, pf)
        d_real = real["MAE"] - base["MAE"]
        uni = metrics(y, np.clip(pr + float(np.mean(dt)), 0.0, None))
        z0 = metrics(y, np.clip(pr + np.where(z == 0, float(np.mean(dt)), 0.0), 0.0, None))
        rng = np.random.default_rng(params["placebo_seed"])
        perm = np.empty(params["n_placebo"])
        for i in range(params["n_placebo"]):
            perm[i] = (
                metrics(y, np.clip(pr + rng.permutation(dt), 0.0, None))["MAE"]
                - base["MAE"]
            )
        prows += [
            {
                "dataset": name,
                "scheme": "Actual channel constraint",
                "MAE_change": d_real,
                "RMSE_change": real["RMSE"] - base["RMSE"],
                "bias_change": real["Bias"] - base["Bias"],
            },
            {
                "dataset": name,
                "scheme": "Placebo1 Uniform uplift",
                "MAE_change": uni["MAE"] - base["MAE"],
                "RMSE_change": uni["RMSE"] - base["RMSE"],
                "bias_change": uni["Bias"] - base["Bias"],
            },
            {
                "dataset": name,
                "scheme": "Placebo3 zone0 uplift",
                "MAE_change": z0["MAE"] - base["MAE"],
                "RMSE_change": z0["RMSE"] - base["RMSE"],
                "bias_change": z0["Bias"] - base["Bias"],
            },
            {
                "dataset": name,
                "scheme": f"Placebo2 Random uplift mean ({params['n_placebo']} trials)",
                "MAE_change": float(perm.mean()),
                "RMSE_change": np.nan,
                "bias_change": np.nan,
            },
            {
                "dataset": name,
                "scheme": "Placebo2 null_MAE_change_P5",
                "MAE_change": float(np.percentile(perm, 5)),
                "RMSE_change": np.nan,
                "bias_change": np.nan,
            },
            {
                "dataset": name,
                "scheme": "Placebo2 null_MAE_change_P95",
                "MAE_change": float(np.percentile(perm, 95)),
                "RMSE_change": np.nan,
                "bias_change": np.nan,
            },
        ]
        data[name].attrs["perm"] = perm
        data[name].attrs["d_real"] = d_real
    pdf_pl = pd.DataFrame(prows)
    pdf_pl.to_csv(params["out_dir"] / "20_placebo_tests.csv", index=False, encoding="utf-8-sig")
    plot_evaluation(data, fd, fz, mdf, params=params)
    zdf.to_csv(params["out_dir"] / "23_zone_diagnostics.csv", index=False, encoding="utf-8-sig")
