import argparse
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import Normalize
from utils import (
    _ls_sorted_frames,
    _ls_sorted_trajs,
    plt_diagnostics_com,
)

from l3es.turbulence import u_and_spectrum_from_ur
from l3es.utils import energy_spectrum


def plt_burnin(path: Path, fig_dir: Path):
    """Plot in an imshow the u and v fields and |vel| of the (2, N, N) tensor."""
    data = np.load(path)
    if data.ndim != 3 or data.shape[0] != 2:
        raise ValueError(f"Expected shape (2, N, N), got {data.shape}.")

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    kwargs = {"origin": "lower", "cmap": "turbo", "vmax": 6}
    fields = [data[0].T, data[1].T, np.linalg.norm(data, axis=0).T]
    axs[0].imshow(fields[0], vmin=-6, **kwargs)
    axs[1].imshow(fields[1], vmin=-6, **kwargs)
    axs[2].imshow(fields[2], vmin=0, **kwargs)

    for ax, lbl, f in zip(axs, ["ux", "uy", "|u|"], fields):
        ax.set_title(f"{lbl} (min={f.min():.2f}, max={f.max():.2f})")
    fig.tight_layout()
    fig.savefig(fig_dir / "burnin.png")
    plt.close()
    print("Finished plt_burnin!")


def load_trajectories_com_2d(
    ds_root: Path, burnin=4500, every_nth_frame=70, max_trajs=None
) -> dict:
    """Load 2D rollout trajectories from h5 checkpoints.

    Returns dict with keys:
        - u: (S, T, Np, dim)
        - r: (S, T, Np, dim)
        - t: (T,)
    """
    traj_dirs = _ls_sorted_trajs(ds_root)
    if max_trajs is not None:
        traj_dirs = traj_dirs[:max_trajs]

    frame_files = _ls_sorted_frames(traj_dirs[0])[burnin::every_nth_frame]
    with h5py.File(frame_files[0], "r") as f0:
        n_points, dim = f0["r"].shape

    trajectories = {
        "u": np.zeros((len(traj_dirs), len(frame_files), n_points, dim)),  # (S,T,N,D)
        "r": np.zeros((len(traj_dirs), len(frame_files), n_points, dim)),  # (S,T,N,D)
    }

    for i, traj_dir in enumerate(traj_dirs):
        files = _ls_sorted_frames(traj_dir)[burnin::every_nth_frame]
        for j, frame in enumerate(files):
            with h5py.File(frame, "r") as f:
                trajectories["u"][i, j] = f["u"][:]
                trajectories["r"][i, j] = f["r"][:]

    with open(traj_dirs[0] / "config.yaml", "r") as f:
        meta = yaml.load(f, Loader=yaml.FullLoader)

    times = np.arange(len(frame_files), dtype=float)
    trajectories["t"] = times * (meta["sim"]["dt"] * meta["sim"]["ckp_freq"])
    return trajectories


def animate_rlt_2d(trajs, idx, fig_dir: Path = Path("figs")) -> None:
    """Animate 2D point cloud rollout with |u| as color."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    u_norm = np.linalg.norm(trajs["u"][idx], axis=-1)  # (S,T,N,D) -> (T,N)
    fig, ax = plt.subplots(figsize=(5, 5))
    norm = Normalize(vmin=0, vmax=6)
    length = trajs["t"].shape[0]

    scatter = ax.scatter(
        trajs["r"][idx, 0, :, 0],
        trajs["r"][idx, 0, :, 1],
        c=u_norm[0],
        cmap="turbo",
        norm=norm,
        s=10,
    )
    fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.048)

    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    title = ax.set_title(f"|u| at t={trajs['t'][0]:.2f}")
    fig.tight_layout()

    def update(frame):
        scatter.set_offsets(trajs["r"][idx, frame])
        scatter.set_array(u_norm[frame])
        title.set_text(f"|u| at t={trajs['t'][frame]:.2f}")
        return (scatter, title)

    anim = animation.FuncAnimation(fig, update, frames=length, interval=100, blit=False)
    anim.save(fig_dir / f"anim_{idx}.gif", writer="pillow", fps=10)
    plt.close()
    print("Finished animate_rlt_2d !")


def plt_burnin_com_vs_grid(traj_dir: Path, fig_dir: Path) -> None:
    """Compare point cloud vs grid representations at burnin step.

    Side-by-side visualization of:
    - Left: scatter plot of point cloud positions colored by velocity magnitude
    - Right: imshow of interpolated grid velocity magnitude
    """

    # Load point cloud data from h5 at burnin step (step 4500 is standard burnin)
    h5_file = traj_dir / "com" / "step_04500.h5"
    grid_file = traj_dir / "u_512_04500_burnin.npy"

    if not h5_file.exists() or not grid_file.exists():
        print(f"Skipping plt_burnin_com_vs_grid: no {h5_file.name} or {grid_file.name}")
        return

    with h5py.File(h5_file, "r") as f:
        r = f["r"][:]  # (N, 2)
        u = f["u"][:]  # (N, 2)

    u_grid = np.load(grid_file)  # shape: (2, Nx, Ny)

    # Compute velocity magnitudes
    u_mag_com = np.linalg.norm(u, axis=-1)
    u_mag_grid = np.linalg.norm(u_grid, axis=0)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    kwargs = {"cmap": "turbo", "vmin": 0, "vmax": 6}

    # Point cloud scatter
    sc0 = axs[0].scatter(r[:, 0], r[:, 1], c=u_mag_com, s=20, **kwargs)
    axs[0].set_xlim(0, 2 * np.pi)
    axs[0].set_ylim(0, 2 * np.pi)
    axs[0].set_aspect("equal", adjustable="box")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_title("Point cloud (COM)")
    fig.colorbar(sc0, ax=axs[0], label="|u|")

    # Grid imshow
    im1 = axs[1].imshow(u_mag_grid.T, origin="lower", **kwargs)
    axs[1].set_xlabel("X (grid index)")
    axs[1].set_ylabel("Y (grid index)")
    axs[1].set_title("Grid interpolation")
    fig.colorbar(im1, ax=axs[1], label="|u|")

    fig.savefig(traj_dir / "burnin_com_vs_grid.png", dpi=150)
    plt.close()
    print("Finished plt_burnin_com_vs_grid !")


def plt_spectrum_burnin_2d(path: Path, max_trajs=10):
    """Plot energy spectra of burn-in grid fields for a few trajectories."""
    path = Path(path)
    burnin_files = []

    if path.is_file():
        burnin_files = [path]
        fig_dir = path.parent
    else:
        traj_dirs = _ls_sorted_trajs(path)[:max_trajs]
        burnin_files = [traj_dir / "u_512_04500_burnin.npy" for traj_dir in traj_dirs]
        fig_dir = path / "figs"

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    n_grid = None
    for file in burnin_files:
        if not file.exists():
            continue
        u = np.load(file)[..., None]
        n_grid = u.shape[1]
        ek = np.asarray(energy_spectrum(u, dim=2))
        k = np.arange(1, len(ek))
        ax.plot(k, ek[1:], label=file.parent.name)

    if n_grid is not None:
        ax.axvline(n_grid // 2, c="tab:orange", ls="--", label="N/2")
        ax.axvline(n_grid // 3, c="tab:green", ls="--", label="N/3")
    ax.plot(k, k ** (-3.0), "--", c="k", label="k**(-3)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("E(k)")
    ax.set_title("Burn-in energy spectra (2D)")
    ax.grid()
    ax.set_ylim(1e-8, 1e1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "spectrum_burnin_2d.png", dpi=200)
    plt.close()
    print("Finished plt_spectrum_burnin_2d !")


def plt_spectrum_points_2d(path: Path, max_trajs=6):
    """Similar to plt_spectrum_burnin_2d, but based on point-cloud data at:
    1. The same burnin step.
    2. At the very last step of the rollout for same trajs.
    Underscore the differences in initial to final spectrum, and across trajectories.
    """
    ds_root = Path(path)
    traj_dirs = _ls_sorted_trajs(ds_root)[:max_trajs]

    fig_dir = ds_root / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    n_grid = None
    k_ref = None
    cmap = plt.get_cmap("tab10")
    L = 2 * np.pi

    for i, traj_dir in enumerate(traj_dirs):
        frames = _ls_sorted_frames(traj_dir)
        if len(frames) < 2:
            continue

        with h5py.File(frames[0], "r") as f0, h5py.File(frames[-1], "r") as f1:
            r0, u0 = f0["r"][:], f0["u"][:]
            r1, u1 = f1["r"][:], f1["u"][:]

        n_grid_guess = round(r0.shape[0] ** (1.0 / 2.0))  # 2D -> sqrt
        n_grid = n_grid_guess
        interpolate_and_spectrum = u_and_spectrum_from_ur(n_grid_guess, L, dim=2)
        _, ek0 = interpolate_and_spectrum(r0, u0)
        _, ek1 = interpolate_and_spectrum(r1, u1)

        ek0 = np.asarray(ek0)
        ek1 = np.asarray(ek1)
        k = np.arange(1, len(ek0))
        k_ref = k

        color = cmap(i % 10)
        ax.plot(k, ek0[1:], c=color, ls="-", label=f"{traj_dir.name} start")
        ax.plot(k, ek1[1:], c=color, ls="--", label=f"{traj_dir.name} end")

    if k_ref is None:
        raise RuntimeError("No valid trajectories with at least 2 frames were found.")

    if n_grid is not None:
        ax.axvline(n_grid // 2, c="tab:orange", ls="--", label="N/2")
        ax.axvline(n_grid // 3, c="tab:green", ls="--", label="N/3")
    ax.plot(k_ref, k_ref ** (-3.0), "--", c="k", label="k**(-3)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("E(k)")
    ax.set_title("Point-cloud spectra: start vs end (2D)")
    ax.grid()
    ax.set_ylim(1e-8, 1e1)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "spectrum_points_2d.png", dpi=200)
    plt.close()
    print("Finished plt_spectrum_points_2d !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect 2D Kolmogorov trajectories.")
    parser.add_argument(
        "--src", type=Path, default=Path("dataset_kolm/raw/2D_KOLM_4096_140kevery1")
    )
    parser.add_argument("--burnin", type=int, default=0, help="Burnin steps to skip")
    parser.add_argument(
        "--every_nth_frame", type=int, default=100, help="Load every nth frame"
    )
    parser.add_argument(
        "--max_trajs", type=int, default=3, help="Max trajectories to plot"
    )
    parser.add_argument(
        "--n_anim", type=int, default=1, help="Number of trajectories to animate"
    )
    args = parser.parse_args()

    fig_dir = args.src / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Plot burnin field visualization
    traj_dirs = _ls_sorted_trajs(args.src)[: args.max_trajs]
    plt_burnin(traj_dirs[0] / "u_512_04500_burnin.npy", traj_dirs[0])

    # Plot point cloud vs grid comparison
    plt_burnin_com_vs_grid(traj_dirs[0], fig_dir)

    # Plot diagnostics evolution
    plt_diagnostics_com(args.src, max_trajs=args.max_trajs)

    # Animate trajectories
    trajs = load_trajectories_com_2d(
        args.src,
        burnin=args.burnin,
        every_nth_frame=args.every_nth_frame,
        max_trajs=args.n_anim,
    )
    for idx in range(len(traj_dirs[: args.n_anim])):
        animate_rlt_2d(trajs, idx=idx, fig_dir=fig_dir)

    # Plot spectra
    plt_spectrum_burnin_2d(args.src, max_trajs=args.max_trajs)
    plt_spectrum_points_2d(args.src, max_trajs=args.max_trajs)
