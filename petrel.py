from pathlib import Path
import re
import numpy as np
import pandas as pd
from tqdm import tqdm


def grid_header(path):
    with path.open(encoding="utf-8-sig") as file:
        header = []
        for line in file:
            if not line.startswith("#"):
                break
            header.append(line)
    text = "".join(header)
    columns, rows = map(int, re.search(r"Grid_size:\s*(\d+)\s*x\s*(\d+)", text).groups())
    bounds = re.search(r"Grid_space:\s*([^\n]+)", text).group(1)
    xmin, xmax, ymin, ymax = map(float, bounds.split(","))
    return columns, rows, xmin, ymin, (xmax - xmin) / (columns - 1), (ymax - ymin) / (rows - 1)


def convert_attribute(path, destination, grid, occupied, chunk_size):
    columns, rows, xmin, ymin, dx, dy = grid
    values = np.lib.format.open_memmap(destination, mode="w+", dtype="float64", shape=(rows * columns,))
    values[:] = np.nan
    seen = np.zeros(rows * columns, dtype=bool)
    with path.open("rb") as stream:
        with tqdm.wrapattr(stream, "read", total=path.stat().st_size,
                desc=path.name, unit="B", unit_scale=True, ascii=True,
                dynamic_ncols=True, position=1, leave=False) as progress:
            reader = pd.read_csv(progress, encoding="utf-8-sig", sep=r"\s+", comment="#", header=None,
                names=["X", "Y", "value", "column", "row"],
                dtype={"column": "int32", "row": "int32"}, chunksize=chunk_size)
            for data in reader:
                # Petrel grid indices are one-based; coordinates remain unchanged.
                column = data["column"].to_numpy() - 1
                row = data["row"].to_numpy() - 1
                if ((column < 0) | (column >= columns) | (row < 0) | (row >= rows)).any():
                    raise ValueError(f"Grid index outside declared dimensions: {path.name}")
                index = row * columns + column
                if seen[index].any() or pd.Index(index).has_duplicates:
                    raise ValueError(f"Duplicate grid point: {path.name}")
                if not (np.allclose(data["X"], xmin + column * dx, atol=1e-6, rtol=0)
                        and np.allclose(data["Y"], ymin + row * dy, atol=1e-6, rtol=0)):
                    raise ValueError(f"Coordinates disagree with grid indices: {path.name}")
                values[index] = data["value"].to_numpy()
                seen[index] = True
                occupied[index] = True
    values.flush()


def run(*, params):
    source = Path(params["source_dir"])
    output = Path(params["output_csv"])
    work = Path(params["work_dir"])
    attributes = params["attributes"]
    output.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    grid = grid_header(source / next(iter(attributes)))
    columns, rows, xmin, ymin, dx, dy = grid
    occupied = np.zeros(rows * columns, dtype=bool)
    temporary = []
    for number, name in enumerate(tqdm(attributes, desc="Converting attributes",
            unit="file", ascii=True, dynamic_ncols=True)):
        if grid_header(source / name) != grid:
            raise ValueError(f"Inconsistent grid header: {name}")
        path = work / f"attribute_{number}.npy"
        convert_attribute(source / name, path, grid, occupied, params["chunk_size"])
        temporary.append(path)
    arrays = [np.load(path, mmap_mode="r") for path in temporary]
    partial = output.with_suffix(".partial.csv")
    first = True
    # Align by grid cell rather than file row; missing attributes stay empty.
    with tqdm(total=int(occupied.sum()), desc="Writing CSV", unit="row",
            unit_scale=True, ascii=True, dynamic_ncols=True) as progress:
        for start in range(0, len(occupied), params["chunk_size"]):
            index = np.flatnonzero(occupied[start:start + params["chunk_size"]]) + start
            if not len(index):
                continue
            data = {"X": xmin + index % columns * dx, "Y": ymin + index // columns * dy}
            data.update({label: values[index] for label, values in zip(attributes.values(), arrays)})
            pd.DataFrame(data).to_csv(partial, mode="w" if first else "a", header=first,
                index=False, encoding="utf-8" if not first else "utf-8-sig", float_format="%.6f")
            first = False
            progress.update(len(index))
    partial.replace(output)
    for values in arrays:
        values._mmap.close()
    for path in temporary:
        path.unlink()
    work.rmdir()
    tqdm.write(f"Saved: {output.resolve()}")
