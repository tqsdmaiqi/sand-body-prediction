from dataset import read_csv
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def rank_features(df, *, params):
    target = "sand_thickness_cn"
    df = df.rename(
        columns={
            "amplitude_std": "std_amplitude",
            "seismic_threshold": "threshold_value",
        }
    )
    data = df[[target, *params["attr_cols"]]].dropna(subset=[target])
    col_map = {name: name for name in params["attr_cols"]}
    rows = []
    for col_orig in params["attr_cols"]:
        col_act = col_map[col_orig]
        x = data[col_act].to_numpy(dtype=float)
        y = data[target].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        pearson = float(pearsonr(x[mask], y[mask])[0])
        rows.append({"attribute": col_orig, "Pearson": pearson})
    out = pd.DataFrame(rows).sort_values("Pearson", key=abs, ascending=False)
    return out


def run(*, params) -> None:
    params["out_dir"].mkdir(parents=True, exist_ok=True)
    df = read_csv(params["input_csv"], options=params["csv_options"])
    out = rank_features(df, params=params)
    out.to_csv(params["out_dir"] / "01_feature_metrics.csv", index=False, encoding="utf-8-sig")
