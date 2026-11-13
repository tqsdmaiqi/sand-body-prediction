from metrics import regression_metrics as plot_metrics
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error


def plot_scatter(
    df: pd.DataFrame, y_col: str, pred_col: str, path: Path, title: str
) -> None:
    y_true = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
    lo = min(np.nanmin(y_true), np.nanmin(y_pred))
    hi = max(np.nanmax(y_true), np.nanmax(y_pred))
    metrics = plot_metrics(y_true, y_pred)
    plt.figure(figsize=(6.6, 5.8))
    plt.scatter(y_true, y_pred, s=44, alpha=0.82)
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel("observed_sand_thickness")
    plt.ylabel("predicted_sand_thickness")
    plt.title(
        f"{title}\nR2={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.3f}, MAE={metrics['MAE']:.3f}"
    )
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_morphology(full_pred, c2_col, *, params):
    xg = pd.to_numeric(full_pred[params["x_col"]], errors="coerce").to_numpy(
        dtype=float
    )
    yg = pd.to_numeric(full_pred[params["y_col"]], errors="coerce").to_numpy(
        dtype=float
    )
    zc2 = pd.to_numeric(full_pred[c2_col], errors="coerce").to_numpy(dtype=float)
    zfin = pd.to_numeric(
        full_pred[params["morph_col_final"]], errors="coerce"
    ).to_numpy(dtype=float)
    zraw = pd.to_numeric(
        full_pred["pred_sand_thickness_raw"], errors="coerce"
    ).to_numpy(dtype=float)
    zd = pd.to_numeric(full_pred[params["morph_col_delta"]], errors="coerce").to_numpy(
        dtype=float
    )
    sc = dict(s=9, marker="s", linewidths=0, rasterized=True)
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 12.6), constrained_layout=True)
    vmax = float(np.nanpercentile(zfin, 99))
    for ax, ttl, z in (
        (axes[0, 0], "(a) C2 channel constraint", zc2),
        (axes[0, 1], "(b) After channel continuity adjustment (final map)", zfin),
        (axes[1, 0], "(c) Raw prediction (reference)", zraw),
    ):
        mm = ax.scatter(xg, yg, c=z, cmap="YlGnBu", vmin=0.0, vmax=vmax, **sc)
        fig.colorbar(mm, ax=ax, shrink=0.82, label="sand_thickness_cn (m)")
        ax.set_title(ttl, fontsize=11)
    mm = axes[1, 1].scatter(
        xg,
        yg,
        c=zd,
        cmap="Reds",
        vmin=0.0,
        vmax=max(float(np.nanpercentile(zd, 99)), 1e-06),
        **sc,
    )
    fig.colorbar(mm, ax=axes[1, 1], shrink=0.82, label="Adjusted - C2 (m)")
    axes[1, 1].set_title(
        f"(d) uplift: {int((zd > 1e-09).sum())} cells, mean {np.nanmean(zd):+.3f} m",
        fontsize=11,
    )
    for ax in axes.ravel():
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(alpha=0.15, lw=0.4)
        ax.ticklabel_format(style="plain", useOffset=False)
        for lb in ax.get_xticklabels():
            lb.set_rotation(30)
            lb.set_ha("right")
    fig.suptitle(
        "Channel continuity adjustment (within-band floor envelope + directional gap bridging)* map postprocessing, model unchanged",
        fontsize=13,
    )
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        p = params["out_dir"] / f"16_morphology_panels.{ext}"
        fig.savefig(p, **kw)
    plt.close(fig)


def plot_ablation(predictions, repeat_metrics, summary, out_dir):
    order = [
        "C0 No channel prior",
        "C1 Sampling prior",
        "C2 Prediction constraint",
        "C3 Both stages",
    ]
    colors = ["#7F7F7F", "#4C72B0", "#55A868", "#C44E52"]
    short = ["C0", "C1", "C2", "C3"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for idx, (metric, ylabel) in enumerate(
        zip(["R2", "RMSE", "MAE"], ["$R^2$", "RMSE (m)", "MAE (m)"])
    ):
        ax = axes[idx]
        dl = [
            repeat_metrics[repeat_metrics["experiment"] == s][metric].values
            for s in order
        ]
        bp = ax.boxplot(
            dl,
            tick_labels=short,
            patch_artist=True,
            widths=0.5,
            showmeans=True,
            meanprops=dict(
                marker="D",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=8,
            ),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(labelsize=11)
    fig.suptitle(
        "Table 5-2 Two-stage channel-prior ablation", fontsize=14, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "figure5_2_metric_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for idx, scheme in enumerate(order):
        ax = axes.flatten()[idx]
        part = predictions[predictions["experiment"] == scheme]
        yt, yp = (
            part["y_true"].to_numpy(dtype=float),
            part["y_pred"].to_numpy(dtype=float),
        )
        r2 = r2_score(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        ax.scatter(yt, yp, c=colors[idx], alpha=0.5, s=15)
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.7)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("observed thickness (m)", fontsize=11)
        ax.set_ylabel("predicted thickness (m)", fontsize=11)
        ax.set_title(f"{scheme}\n$R^2$={r2:.4f}, RMSE={rmse:.2f}m", fontsize=11)
        ax.set_aspect("equal")
    fig.suptitle(
        "Table 5-2 OOF predictions vs observations", fontsize=14, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(
        out_dir / "figure5_2_prediction_scatter.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def plot_evaluation(data, fd, fz, mdf, *, params):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11), constrained_layout=True)
    ax = axes[0, 0]
    for z, c in ((0, "#888888"), (1, "#1f77b4"), (2, "#d62728")):
        v = fd[fz == z]
        v = v[np.abs(v) > 1e-09]
        if len(v):
            ax.hist(
                v,
                bins=40,
                histtype="step",
                lw=1.8,
                color=c,
                label=f"{params['zone_name'][z]} ({len(v)} cells)",
            )
    ax.set_yscale("log")
    ax.set_xlabel("deltaT = constrained - raw (m)")
    ax.set_ylabel("Cell count (log)")
    ax.set_title("(a) Adjustment by channel zone\nzone0 unchanged")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax = axes[0, 1]
    for name, mk in zip(data, ("o", "s", "^")):
        d = data[name]
        ax.scatter(
            d[params["col_raw"]],
            d[params["col_fin"]],
            s=26,
            marker=mk,
            alpha=0.75,
            label=name,
        )
    lo = min(
        float(np.nanmin([d[params["col_raw"]].min() for d in data.values()])),
        float(np.nanmin([d[params["col_fin"]].min() for d in data.values()])),
    )
    hi = max(
        float(np.nanmax([d[params["col_raw"]].max() for d in data.values()])),
        float(np.nanmax([d[params["col_fin"]].max() for d in data.values()])),
    )
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Raw prediction (m)")
    ax.set_ylabel("Constrained prediction (m)")
    ax.set_title(
        "(b) Raw versus constrained well predictions\nPoints above the identity line indicate uplift"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax = axes[1, 0]
    labels = [n for n, _ in params["datasets"]]
    x = np.arange(len(labels))
    w = 0.35
    a_mae = [
        plot_metrics(data[n][params["y_col"]], data[n][params["col_raw"]])["MAE"]
        for n in labels
    ]
    b_mae = [
        plot_metrics(data[n][params["y_col"]], data[n][params["col_fin"]])["MAE"]
        for n in labels
    ]
    ax.bar(x - w / 2, a_mae, w, label="raw", color="#9ecae1")
    ax.bar(x + w / 2, b_mae, w, label="constrained", color="#e6550d")
    for i, (a, b) in enumerate(zip(a_mae, b_mae)):
        ax.text(i - w / 2, a + 0.02, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + w / 2, b + 0.02, f"{b:.3f}", ha="center", fontsize=9)
        p = mdf.loc[mdf["dataset"] == labels[i], "well_paired_t_p"]
        pv = float(p.iloc[0]) if len(p) and np.isfinite(p.iloc[0]) else np.nan
        ax.text(
            i,
            max(a, b) + 0.18,
            f"well-level p={pv:.3f}" if np.isfinite(pv) else "",
            ha="center",
            fontsize=9,
            color="#333333",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(n={len(data[n])})" for n in labels])
    ax.set_ylabel("MAE (m)")
    ax.set_title("(c) Well MAE before and after constraints")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    ax = axes[1, 1]
    for name, c in zip(data, ("#1f77b4", "#2ca02c", "#d62728")):
        perm = data[name].attrs.get("perm")
        if perm is None:
            continue
        ax.hist(
            perm,
            bins=30,
            alpha=0.45,
            color=c,
            label=f"{name} random-location null distribution",
        )
        ax.axvline(data[name].attrs["d_real"], color=c, lw=2.4, ls="--")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("MAE change = constrained - raw (m; lower is better)")
    ax.set_ylabel("iterations")
    ax.set_title(
        "(d) Random-location uplift null distribution\nDashed line: actual constraint"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.suptitle(
        "C2 constraint evaluation: uplift-only adjustment and channel topology",
        fontsize=13,
    )
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        p = params["out_dir"] / f"22_constraint_evaluation.{ext}"
        fig.savefig(p, **kw)
    plt.close(fig)
