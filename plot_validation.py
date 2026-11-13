from pathlib import Path
from dataset import read_csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def find_file(name: str, *, params) -> Path:
    return params["channel_output"] / name


def load_pair(path: Path, *, params) -> tuple[np.ndarray, np.ndarray]:
    data = read_csv(path, options=params["csv_options"])
    missing = [
        c for c in (params["observed_col"], params["pred_col"]) if c not in data.columns
    ]
    if missing:
        raise KeyError(
            f"{path.name} Missing columns: {missing}\nAvailable columns: {data.columns.tolist()}"
        )
    obs = pd.to_numeric(data[params["observed_col"]], errors="coerce").to_numpy(float)
    pred = pd.to_numeric(data[params["pred_col"]], errors="coerce").to_numpy(float)
    keep = np.isfinite(obs) & np.isfinite(pred)
    return (obs[keep], pred[keep])


def metrics(obs: np.ndarray, pred: np.ndarray, *, params) -> dict[str, float]:
    residual = pred - obs
    ss_total = np.sum((obs - obs.mean()) ** 2)
    r = float(np.corrcoef(obs, pred)[0, 1]) if len(obs) > 1 else np.nan
    return {
        "n": len(obs),
        "r": r,
        "R2_sse": (float(1.0 - np.sum(residual**2) / ss_total) if ss_total > 0 else np.nan),
        "R2_pearson2": float(r**2),
        "R2": (
            float(1.0 - np.sum(residual**2) / ss_total)
            if ss_total > 0 and params["r2_def"] == "sse"
            else float(r**2)
        ),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAE": float(np.mean(np.abs(residual))),
        "Bias": float(np.mean(residual)),
    }


def run(*, params):
    params["out_dir"].mkdir(parents=True, exist_ok=True)
    SERIES = []
    for key, label, fname, color in (
        ("train", "Training", params["train_name"], params["c_train"]),
        ("test", "Test", params["test_name"], params["c_test"]),
        ("valid", "Validation", params["valid_name"], params["c_valid"]),
    ):
        path = find_file(fname, params=params)
        obs, pred = load_pair(path, params=params)
        SERIES.append(
            {
                "key": key,
                "label": label,
                "obs": obs,
                "pred": pred,
                "color": color,
                "file": path,
                "m": metrics(obs, pred, params=params),
            }
        )
    PANEL_A = [d for d in SERIES if d["key"] in ("train", "test")]
    PANEL_B = [d for d in SERIES if d["key"] == "valid"]
    PANELS = (PANEL_A, PANEL_B)
    if params["shared_axis"]:
        span_max = float(max((max(d["obs"].max(), d["pred"].max()) for d in SERIES)))
        axis_max = float(np.ceil(span_max / params["tick_step"]) * params["tick_step"])
        if span_max >= axis_max - 0.2:
            axis_max += params["tick_step"]
        common = (0.0, axis_max)
    else:
        common = None
    for group in PANELS:
        for d in group:
            if common is not None:
                lims = common
            else:
                sm = float(max(d["obs"].max(), d["pred"].max()))
                am = float(np.ceil(sm / params["tick_step"]) * params["tick_step"])
                if sm >= am - 0.2:
                    am += params["tick_step"]
                lims = (0.0, am)
            d["lims"] = lims
            lo, hi = lims
            assert (
                d["obs"].min() >= lo and d["obs"].max() <= hi
            ), f"{d['label']} Observed values outside limits"
            assert (
                d["pred"].min() >= lo and d["pred"].max() <= hi
            ), f"{d['label']} Predicted values outside limits"

    def block_of(d: dict) -> str:
        m = d["m"]
        parts = []
        if params["show_r"]:
            parts.append(f"$r$={m['r']:.2f}")
        if params["show_r2"]:
            parts.append(f"$R^2$={m['R2']:.2f}")
        if params["show_rmse"]:
            parts.append(f"RMSE={m['RMSE']:.2f} m")
        if params["show_mae"]:
            parts.append(f"MAE={m['MAE']:.2f} m")
        return f"{d['label']} ($n$={m['n']})\n   " + "  ".join(parts)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0), constrained_layout=True)
    for ax, group, plabel in zip(axes, PANELS, params["panel_labels"]):
        for d in group:
            ax.scatter(
                d["obs"],
                d["pred"],
                s=params["marker_size"],
                c=d["color"],
                alpha=params["alpha"],
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
                label=d["label"],
            )
        lims = group[0]["lims"]
        ax.plot(
            lims,
            lims,
            params["line_style"],
            color=params["c_line"],
            linewidth=params["line_width"],
            zorder=2,
            label="1:1 line",
        )
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Observed sandbody thickness (m)", fontsize=params["label_fontsize"])
        ax.set_ylabel("Predicted sandbody thickness (m)", fontsize=params["label_fontsize"])
        if params["tick_step"] > 0:
            ticks = np.arange(lims[0], lims[1] + 0.1, params["tick_step"])
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if params["show_panel_labels"]:
            ax.set_title(plabel, loc="left", fontsize=11.5, fontweight="bold", pad=4)
        if params["show_legend"]:
            handles, labels = ax.get_legend_handles_labels()
            keep = [(h, l) for h, l in zip(handles, labels) if l != "1:1 line"]
            leg = ax.legend(
                [h for h, _ in keep],
                [l for _, l in keep],
                loc="lower right",
                frameon=False,
                fontsize=9.5,
                handletextpad=0.35,
                borderpad=0.2,
                labelspacing=0.35,
            )
            for h in leg.legend_handles:
                h.set_alpha(1.0)
        ax.text(
            0.05,
            0.95,
            "\n".join((block_of(d) for d in group)),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            linespacing=1.5,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="none",
                linewidth=0.0,
                alpha=0.92,
            ),
        )
    for ext, kwargs in [
        ("png", {"dpi": params["dpi"], "bbox_inches": "tight"}),
        ("pdf", {"bbox_inches": "tight"}),
        ("svg", {"bbox_inches": "tight"}),
    ]:
        output = params["out_dir"] / f"{params['out_stem']}.{ext}"
        fig.savefig(output, **kwargs)
    plt.close(fig)
