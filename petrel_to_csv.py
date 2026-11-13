"""Integrate Petrel scattered-data attribute exports into one grid CSV.

Usage:
    python petrel_to_csv.py <source_dir> <output_csv>
"""

import argparse
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

FILES = [
    ("7arclength", "arc_length"),
    ("7average energy", "average_energy"),
    ("7average instantaneous frequency", "average_instantaneous_frequency"),
    ("7average instantaneous phase", "average_instantaneous_phase"),
    ("7average negative amplitude", "average_negative_amplitude"),
    ("7half energy", "half_energy"),
    ("7maximum amplitude", "maximum_amplitude"),
    ("7mean amplitude", "mean_amplitude"),
    ("7minimum amplitude", "minimum_amplitude"),
    ("7RMS", "rms_amplitude"),
    ("7standard deviation of amplitude", "std_amplitude"),
    ("7sum of amplitude", "amplitude_sum"),
    ("7sum of energy", "total_energy"),
    ("7sum of negative amplitude", "negative_amplitude_sum"),
    ("7threshold value", "threshold_value"),
]


def grid_header(path):
    with open(path, "rb") as file:
        head = []
        for line in file:
            if not line.startswith(b"#"):
                break
            head.append(line)
    text = b"".join(head).decode("utf-8", "replace")
    columns, rows = map(int, re.search(r"Grid_size:\s*(\d+)\s*x\s*(\d+)", text).groups())
    xmin, xmax, ymin, ymax = map(float, re.search(r"Grid_space:\s*([^\n]+)", text).group(1).split(","))
    return columns, rows, xmin, ymin, (xmax - xmin) / (columns - 1), (ymax - ymin) / (rows - 1)


def read_points(path):
    x, y, value, column, row = [], [], [], [], []
    with open(path, "rb") as file:
        for line in file:
            if line[:1] == b"#":
                continue
            parts = line.split()
            x.append(float(parts[0]))
            y.append(float(parts[1]))
            value.append(float(parts[2]))
            column.append(int(parts[3]))
            row.append(int(parts[4]))
    return (np.array(x), np.array(y), np.array(value),
            np.fromiter(column, np.int32, len(column)), np.fromiter(row, np.int32, len(row)))


def load_attribute(path, grid, values, occupied):
    columns, rows, xmin, ymin, dx, dy = grid
    x, y, value, column, row = read_points(path)
    if ((column < 1) | (column > columns) | (row < 1) | (row > rows)).any():
        raise ValueError(f"Grid index outside declared extent: {path.name}")
    if not (np.allclose(x, xmin + (column - 1) * dx, atol=1e-3, rtol=0)
            and np.allclose(y, ymin + (row - 1) * dy, atol=1e-3, rtol=0)):
        raise ValueError(f"Coordinates disagree with grid indices: {path.name}")
    index = (row - 1) * columns + (column - 1)
    if np.unique(index).size != index.size:
        raise ValueError(f"Duplicate grid point: {path.name}")
    values[index] = value
    occupied[index] = True
    return index.size, float(value.min()), float(value.max())


def run(source, output, chunk=200000):
    grid = grid_header(source / FILES[0][0])
    for name, _ in FILES[1:]:
        if grid_header(source / name) != grid:
            raise ValueError(f"Inconsistent grid header: {name}")
    columns, rows = grid[0], grid[1]
    cells = rows * columns
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="petrel_"))
    try:
        arrays = [np.lib.format.open_memmap(work / f"attribute_{number}.npy", mode="w+",
                                            dtype=np.float64, shape=(cells,))
                  for number in range(len(FILES))]
        for values in arrays:
            values[:] = np.nan
        occupied = np.zeros(cells, dtype=bool)
        for number, (name, label) in enumerate(FILES):
            count, low, high = load_attribute(source / name, grid, arrays[number], occupied)
            print(f"{label}: {count} points, range [{low:.6g}, {high:.6g}]", flush=True)
        partial = output.with_suffix(".partial.csv")
        partial.unlink(missing_ok=True)
        first = True
        for start in range(0, cells, chunk):
            index = np.flatnonzero(occupied[start:start + chunk]) + start
            if not index.size:
                continue
            data = {"X": grid[2] + index % columns * grid[4],
                    "Y": grid[3] + index // columns * grid[5]}
            for label, values in zip([label for _, label in FILES], arrays):
                data[label] = values[index]
            pd.DataFrame(data).to_csv(partial, mode="w" if first else "a", header=first,
                index=False, encoding="utf-8-sig" if first else "utf-8", float_format="%.6f")
            first = False
            print(f"written {min(start + chunk, cells)} / {cells}", flush=True)
        partial.replace(output)
        return int(occupied.sum())
    finally:
        del arrays
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir")
    parser.add_argument("output_csv")
    args = parser.parse_args()
    count = run(Path(args.source_dir), Path(args.output_csv))
    print(f"total grid cells with data: {count}")
