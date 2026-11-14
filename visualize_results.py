"""Compare Petrel predictions; display <4 m as zero and cap >18 m."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="Open the figure window")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    files = [
        ("coarse_grid_sand_thickness_scattered_data", "BRT prediction"),
        (
            "scattered_sand_thickness_channel_constrained",
            "Channel-constrained prediction",
        ),
    ]
    cmap = plt.get_cmap("turbo").with_extremes(under="white", bad="#eeeeee")
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5.5), layout="constrained", sharex=True, sharey=True
    )
    for ax, (name, title) in zip(axes, files):
        data = pd.read_csv(
            root / "outputs" / name,
            comment="#",
            sep=r"\s+",
            header=None,
            names=["x", "y", "z", "column", "row"],
        )
        data["z"] = np.where(data.z < 4, 0, np.minimum(data.z, 18))
        image = (
            data.pivot(index="y", columns="x", values="z")
            .sort_index()
            .sort_index(axis=1)
        )
        mesh = ax.pcolormesh(
            image.columns.to_numpy() / 1000,
            image.index.to_numpy() / 1000,
            image.to_numpy(),
            shading="nearest",
            cmap=cmap,
            vmin=4,
            vmax=18,
            rasterized=True,
        )
        ax.set(title=title, xlabel="Easting (km)", aspect="equal")
        ax.ticklabel_format(style="plain", useOffset=False)
    axes[0].set_ylabel("Northing (km)")
    fig.colorbar(
        mesh,
        ax=axes,
        label="Sand thickness (m)",
        ticks=np.arange(4, 19, 2),
        shrink=0.85,
    )
    fig.supxlabel(
        "White: <4 m (displayed as 0); gray: outside available grid", fontsize=9
    )
    output = root / "figures" / "prediction_comparison.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=300)
    print(output)
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
