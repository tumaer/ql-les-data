"""Convert a custom simulation to a NetCDF file compatible with xarray.
The requirement is that the simulation is ran using main.py with mode=simulate.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import xarray
import yaml


def custom_dataset_to_netcdf(dataset_root: Path, L=2 * np.pi, Nx=64) -> None:
    """Converts single simulation trajectory to NetCDF file compatible with xarray."""
    with open(dataset_root / "config.yaml", "r") as f:
        meta = yaml.load(f, Loader=yaml.FullLoader)

    n_times = len(os.listdir(dataset_root / "sim"))
    u = np.zeros((2, 1, n_times, Nx, Nx))  # (Dim, Trajs, Time, X, Y)

    files = os.listdir(dataset_root / "sim")
    files.sort()
    files = files
    for j, f in enumerate(files):
        u[:, 0, j] = np.load(dataset_root / "sim" / f)  # (2,64,64)

    time = np.arange(len(files)) * meta["sim"]["dt"] * meta["sim"]["ckp_freq"]
    x = y = (np.arange(Nx) + 0.5) * L / Nx

    ds = xarray.Dataset(
        data_vars={
            "u": (("sample", "time", "x", "y"), u[0]),
            "v": (("sample", "time", "x", "y"), u[1]),
        },
        coords={"sample": np.arange(1), "time": time, "x": x, "y": y},
        attrs={"description": "Synthetic dataset for testing"},
    )
    print(f"Dataset: \n{ds}")
    ds.to_netcdf(dataset_root / "trajs.nc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("results/kolm64_1"))
    args = parser.parse_args()

    custom_dataset_to_netcdf(args.src)
