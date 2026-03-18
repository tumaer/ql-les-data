"""Download public datasets from gresearch and analyze them.

This script can also be applied to our own data generated with:
`python main.py config=configs/kolm64_1.yaml mode=simulate`
and then converted to NetCDF with:
`python scripts/np2xarray.py --src=results/kolm64_1`
Then just run `python scripts/inspect_jaxcfd.py --src=results/kolm64_1/trajs.nc`

Get validation data with:
gsutil ls 'gs://gresearch/jax-cfd/*'
gsutil -m cp gs://gresearch/jax-cfd/public_eval_datasets/kolmogorov_re_1000/eval_64x64_64x64.nc content
gsutil -m cp gs://gresearch/jax-cfd/public_eval_datasets/kolmogorov_re_1000/eval_2048x2048_64x64.nc content
gsutil -m cp gs://gresearch/jax-cfd/public_eval_datasets/kolmogorov_re_1000/long_eval_2048x2048_64x64.nc content
gsutil -m cp gs://gresearch/jax-cfd/public_eval_datasets/decaying/eval_2048x2048_64x64.nc content_decaying
"""  # noqa: E501

import argparse
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import jax_cfd.data as cfd_data
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import xarray


def xarray_open(path: Path) -> xarray.Dataset:
    """Opens a NetCDF dataset using the notebook chunking pattern."""
    return xarray.open_dataset(path, chunks={"time": "100MB"})


def print_dataset_shapes(path: Path) -> None:
    """Prints dataset dimensions and per-variable shapes."""
    ds = xarray_open(path)
    print(f"Dataset: {path}, \n{ds}")
    # for key, value in ds.attrs.items():
    #     if type(value) is str:
    #         print(f"{key}: {type(value)}")
    #     else:
    #         print(f"{key}: {value}")
    # print(ds.data_vars)


def plt_u_max_per_timestep(ds: xarray.Dataset, fig_dir: Path = Path("figs")) -> None:
    """Prints max velocity u at each time step."""
    if "u" not in ds.data_vars:
        raise KeyError("Dataset does not contain variable 'u'.")

    u = ds["u"]
    if "time" not in u.dims:
        raise ValueError("Variable 'u' must contain a 'time' dimension.")

    # reduce_dims = [dim for dim in u.dims if dim != "time"]
    reduce_dims = ["x", "y"]  # Assuming 2D spatial dimensions named 'x' and 'y'
    print(f"Reducing dimensions {reduce_dims} to find max u at each time step.")
    u_max_by_time = u.max(dim=reduce_dims)

    print("Max u at each time step:")

    plt.figure(figsize=(8, 4))
    for idx in range(min(10, u_max_by_time.sizes.get("sample", 1))):
        plt.plot(
            u_max_by_time.sel(sample=idx)["time"].values,
            u_max_by_time.isel(sample=idx).values,
        )
    plt.title("Max u vs Time")
    plt.xlabel("Time")
    plt.ylabel("Max u")
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "max_u_over_time.png")
    plt.close()


def plot_vorticity_frames(
    trajectory: xarray.Dataset,
    num_frames: int = 5,
    sample_index: int = 0,
    fig_dir: Path = Path("figs"),
    frames="jaxcfd_paper",
) -> None:
    """Plots equidistant vorticity frames from a trajectory dataset.

    Args:
        trajectory: Dataset with velocity fields over time (e.g. variables u, v).
        num_frames: Number of equidistant frames to show.
        sample_index: Sample index to plot when a sample dimension exists.
    """
    if "time" not in trajectory.sizes:
        raise ValueError("Trajectory dataset must contain a 'time' dimension.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")

    vorticity_out = cfd_data.xarray_utils.vorticity_2d(trajectory)
    if isinstance(vorticity_out, xarray.DataArray):
        vorticity = vorticity_out
    elif isinstance(vorticity_out, xarray.Dataset):
        if "vorticity" in vorticity_out.data_vars:
            vorticity = vorticity_out["vorticity"]
        else:
            # Fall back to the first variable if naming differs.
            vorticity = next(iter(vorticity_out.data_vars.values()))
    else:
        raise TypeError("vorticity_2d output must be an xarray DataArray or Dataset.")
    if "sample" in vorticity.sizes:
        vorticity = vorticity.isel(sample=sample_index)

    total_steps = vorticity.sizes["time"]
    frame_count = min(num_frames, total_steps)
    if frames == "jaxcfd_paper":
        frame_indices = np.array([0, 80, 160, 210, total_steps - 1])
    else:
        frame_indices = np.linspace(0, total_steps - 1, num=frame_count, dtype=int)
    # selected = vorticity.isel(time=frame_indices)

    fig, axes = plt.subplots(1, frame_count, figsize=(3 * frame_count, 3.5))
    if frame_count == 1:
        axes = [axes]

    # Use shared symmetric bounds so colors are comparable across frames.
    # max_abs = float(np.nanmax(np.abs(selected.values)))
    # vmin, vmax = -max_abs, max_abs
    vmin, vmax = -10, 10

    im = None
    for ax, time_index in zip(axes, frame_indices):
        frame = vorticity.isel(time=int(time_index))
        im = frame.plot.imshow(
            ax=ax,
            cmap=seaborn.cm.icefire,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
        )
        time_value = (
            frame.coords["time"].item() if "time" in frame.coords else int(time_index)
        )
        ax.set_title(f"t={time_value:.2f}")
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Vorticity Frames ({frame_count} Equidistant Time Steps)")
    fig.colorbar(im, ax=axes, shrink=0.95)
    plt.tight_layout()
    plt.savefig(fig_dir / f"vorticity_frames_{sample_index}.png")
    plt.close()


def plt_ekin_evolution(
    trajectory: xarray.Dataset, fig_dir: Path = Path("figs")
) -> None:
    """Plots kinetic energy evolution over time for a given trajectory."""
    if "time" not in trajectory.sizes:
        raise ValueError("Trajectory dataset must contain a 'time' dimension.")
    if "u" not in trajectory.data_vars or "v" not in trajectory.data_vars:
        raise ValueError(
            "Trajectory dataset must contain velocity variables 'u' and 'v'."
        )

    u = trajectory["u"]
    v = trajectory["v"]

    # Compute kinetic energy per unit mass: KE = 0.5 * (u^2 + v^2)
    ke = 0.5 * (u**2 + v**2)
    ke_mean_over_space = ke.mean(dim=["x", "y"])

    plt.figure(figsize=(8, 4))
    for idx in range(min(10, ke_mean_over_space.sizes.get("sample", 1))):
        plt.plot(
            ke_mean_over_space.sel(sample=idx)["time"].values,
            ke_mean_over_space.isel(sample=idx).values,
        )
    # plt.plot(ke_mean_over_space.coords["time"].values, ke_mean_over_space.values)
    plt.title("Kinetic Energy Evolution Over Time")
    plt.xlabel("Time")
    plt.ylabel("Mean Kinetic Energy per Unit Mass")
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "kinetic_energy_evolution.png")
    plt.close()


def plot_kinetic_energy_spectrum_at_time(
    trajectory: xarray.Dataset, time_index: int = 200, fig_dir: Path = Path("figs")
) -> None:
    """Plots isotropic kinetic energy spectrum at a given time index."""
    if "time" not in trajectory.sizes:
        raise ValueError("Trajectory dataset must contain a 'time' dimension.")
    if "u" not in trajectory.data_vars or "v" not in trajectory.data_vars:
        raise ValueError(
            "Trajectory dataset must contain velocity variables 'u' and 'v'."
        )

    total_steps = trajectory.sizes["time"]
    if total_steps == 0:
        raise ValueError("Trajectory dataset has no time steps.")

    time_idx = min(max(time_index, 0), total_steps - 1)
    snapshot = trajectory.isel(time=time_idx)

    time_value = trajectory["time"].isel(time=time_idx).item()
    plt.figure(figsize=(6, 4))
    for idx in range(min(10, trajectory.sizes.get("sample", 1))):
        snapshot_i = snapshot.isel(sample=idx)
        spectrum = cfd_data.xarray_utils.isotropic_energy_spectrum(snapshot_i).compute()
        k_dim = "k" if "k" in spectrum.dims else spectrum.dims[0]
        plt.loglog(spectrum[k_dim].values, spectrum.values, linewidth=2)
    plt.title(f"Kinetic Energy Spectrum at t={time_value:.2f}")
    plt.xlabel("Wavenumber k")
    plt.ylabel("E(k)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"kinetic_energy_spectrum_t{time_idx}.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src", type=Path, default=Path("content_decaying/eval_2048x2048_64x64.nc")
    )
    args = parser.parse_args()

    fig_dir = args.src.parent / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print_dataset_shapes(args.src)
    ds = xarray_open(args.src)
    plt_u_max_per_timestep(ds, fig_dir=fig_dir)
    for i in range(1):  # up to number of trajectories in the dataset
        plot_vorticity_frames(ds, num_frames=5, sample_index=i, fig_dir=fig_dir)
    plt_ekin_evolution(ds, fig_dir=fig_dir)
    plot_kinetic_energy_spectrum_at_time(ds, time_index=0, fig_dir=fig_dir)
    plot_kinetic_energy_spectrum_at_time(ds, time_index=99999, fig_dir=fig_dir)
