from dataset import read_csv
import numpy as np
import pandas as pd


def run(*, params):
    params["out"].mkdir(parents=True, exist_ok=True)
    df = read_csv(params["src"], options=params["csv_options"])
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    rows = []
    for m in params["metrics"]:
        v = pd.to_numeric(df[m], errors="coerce").dropna().to_numpy(float)
        rows.append(
            {
                "metric": m,
                "sample_count": int(v.size),
                "minimum": v.min(),
                "maximum": v.max(),
                "mean": v.mean(),
                "median": float(np.median(v)),
                "standard_deviation": v.std(ddof=1),
                "P05": np.percentile(v, 5),
                "P25": np.percentile(v, 25),
                "P75": np.percentile(v, 75),
                "P95": np.percentile(v, 95),
            }
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(params["out"] / "01_statistics.csv", index=False, encoding="utf-8-sig")
    for i, m in enumerate(params["metrics"], start=2):
        v = pd.to_numeric(df[m], errors="coerce").dropna().to_numpy(float)
        if m == params["metrics"][0]:
            edges = np.logspace(np.log10(v.min()), np.log10(v.max()), params["nbin"] + 1)
        else:
            edges = np.linspace(v.min(), v.max(), params["nbin"] + 1)
        cnt, _ = np.histogram(v, bins=edges)
        out = pd.DataFrame(
            {
                "left_edge": edges[:-1],
                "right_edge": edges[1:],
                "count": cnt,
                "fraction": cnt / v.size,
            }
        )
        f = params["out"] / ("%02d_histogram_%s.csv" % (i, params["label"][m]))
        out.to_csv(f, index=False, encoding="utf-8-sig")
