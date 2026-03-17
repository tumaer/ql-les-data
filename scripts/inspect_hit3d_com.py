"""Visualizations for data generated with:
`python main.py config=configs/hit32_1.yaml`.
"""

import argparse
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from utils import _ls_sorted_frames, _ls_sorted_trajs, load_trajectories_com

from l3es.turbulence import u_and_spectrum_from_ur
from l3es.utils import energy_spectrum


def plt_diagnostics(ds_root, max_trajs=10, fig_dir=Path("figs")):
    """Plot the evolution of a given quantity."""

    traj_dirs = _ls_sorted_trajs(ds_root)
    (ds_root / fig_dir).mkdir(parents=True, exist_ok=True)
    for key in ["e_inj", "ekin", "umax", "rho_max"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        for traj_dir in traj_dirs[:max_trajs]:
            df = pd.read_csv(traj_dir / "diagnostics.csv")
            ax.plot(df["time"], df[key])
        ax.set_xlabel("Time [-]")
        ax.set_ylabel(f"{key}")
        ax.grid()
        fig.tight_layout()
        fig.savefig(ds_root / fig_dir / f"evolution_{key}.png")
        plt.close()
    print("Finished plt_diagnostics !")


def plot_visible_planes_3d(
    ax,
    field,
    vmin,
    vmax,
    title,
    elev=25.0,
    azim=-35,
    roll=0,
):
    """Plot three visible boundary planes of a 3D scalar field on a single axis."""
    n = field.shape[0]
    L = 2 * np.pi
    coords = np.linspace(0.0, L, n)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("turbo")

    yy, zz = np.meshgrid(coords, coords, indexing="ij")
    xx = np.full_like(yy, L)
    ax.plot_surface(
        xx,
        yy,
        zz,
        facecolors=cmap(norm(field[-1, :, :])),
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    xx, zz = np.meshgrid(coords, coords, indexing="ij")
    yy = np.zeros_like(xx)
    ax.plot_surface(
        xx,
        yy,
        zz,
        facecolors=cmap(norm(field[:, 0, :])),
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    zz = np.full_like(xx, L)
    ax.plot_surface(
        xx,
        yy,
        zz,
        facecolors=cmap(norm(field[:, :, -1])),
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(0, L)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{title} (min={field.min():.2f}, max={field.max():.2f})")


def plt_views_burnin_3d(path: Path, fig_dir: Path):
    """Plot 3D visible boundary planes for ux, uy and |u| at DNS-view angle."""
    data = np.load(path)
    if data.ndim != 4 or data.shape[0] != 3:
        raise ValueError(f"Expected shape (3, N, N, N), got {data.shape}.")

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), subplot_kw={"projection": "3d"})
    fields = [data[0], data[1], np.linalg.norm(data, axis=0)]
    labels = ["ux", "uy", "|u|"]
    mins = [-6, -6, 0]
    maxs = [6, 6, 6]
    for ax, field, lbl, vmin, vmax in zip(axs, fields, labels, mins, maxs):
        plot_visible_planes_3d(ax, field, vmin, vmax, lbl)

    fig.tight_layout()
    fig.savefig(fig_dir / "burnin.png")
    plt.close()
    print("Finished plt_views_burnin_3d !")


def animate_rlt_3d(trajs, idx, fig_dir: Path = Path("figs")) -> None:
    """Animate 3D point cloud rollout with |u| as color."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    u_norm = np.linalg.norm(trajs["u"][idx], axis=-1)  # (S,T,N,D) -> (T,N)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    norm = Normalize(vmin=0, vmax=4)
    length = trajs["t"].shape[0]

    scatter = ax.scatter(
        trajs["r"][idx, 0, :, 0],
        trajs["r"][idx, 0, :, 1],
        trajs["r"][idx, 0, :, 2],
        c=u_norm[0],
        cmap="turbo",
        norm=norm,
        s=20,
    )
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(scatter, cax=cax)

    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_zlim(0, 2 * np.pi)
    ax.view_init(elev=25.0, azim=-35, roll=0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title = ax.set_title(f"|u| at t={trajs['t'][0]:.2f}")

    def update(frame):
        xyz = trajs["r"][idx, frame]
        scatter._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
        scatter.set_array(u_norm[frame])
        title.set_text(f"|u| at t={trajs['t'][frame]:.2f}")
        return (scatter, title)

    anim = animation.FuncAnimation(fig, update, frames=length, interval=100, blit=False)
    anim.save(fig_dir / f"anim_{idx}.gif", writer="pillow", fps=10)
    plt.close()
    print("Finished animate_rlt_3d !")


def plt_spectrum_burnin_3d(path: Path, max_trajs=8):
    """Plot energy spectra of burn-in grid fields for a few trajectories."""
    path = Path(path)
    burnin_files = []

    if path.is_file():
        burnin_files = [path]
        fig_dir = path.parent
    else:
        traj_dirs = _ls_sorted_trajs(path)[:max_trajs]
        burnin_files = [traj_dir / "u_256_05000_burnin.npy" for traj_dir in traj_dirs]
        fig_dir = path / "figs"

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    n_grid = None
    for file in burnin_files:
        u = np.load(file)
        n_grid = u.shape[1]
        ek = np.asarray(energy_spectrum(u, dim=3))
        k = np.arange(1, len(ek))
        ax.plot(k, ek[1:], label=file.parent.name)

    if n_grid is not None:
        ax.axvline(n_grid // 2, c="tab:orange", ls="--", label="N/2")
        ax.axvline(n_grid // 3, c="tab:green", ls="--", label="N/3")
    ax.plot(k, k ** (-5 / 3), "--", c="k", label="k**(-5/3)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("E(k)")
    ax.set_title("Burn-in energy spectra (3D)")
    ax.grid()
    ax.set_ylim(1e-8, 1e1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "spectrum_burnin_3d.png", dpi=200)
    plt.close()
    print("Finished plt_spectrum_burnin_3d !")


def plt_spectrum_points_3d(path: Path, max_trajs=6):
    """Similar to plt_spectrum_burnin_3d, but based on point-cloud data at:
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

        n_grid_guess = round(r0.shape[0] ** (1.0 / 3.0))
        n_grid = n_grid_guess
        interpolate_and_spectrum = u_and_spectrum_from_ur(n_grid_guess, L, dim=3)
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
    ax.plot(k_ref, k_ref ** (-5 / 3), "--", c="k", label="k**(-5/3)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("E(k)")
    ax.set_title("Point-cloud spectra: start vs end (3D)")
    ax.grid()
    ax.set_ylim(1e-8, 1e1)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "spectrum_points_3d.png", dpi=200)
    plt.close()
    print("Finished plt_spectrum_points_3d !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir", default=Path("dataset_hit/raw/3D_HIT_32768_20kevery1"), type=Path
    )
    parser.add_argument("--max_trajs", default=3, type=int, help="Max trajs to plot")
    args = parser.parse_args()

    plt_diagnostics(args.src_dir, max_trajs=args.max_trajs, fig_dir=Path("figs"))

    traj_dirs = _ls_sorted_trajs(args.src_dir)[: args.max_trajs]
    for traj_dir in traj_dirs:
        plt_views_burnin_3d(traj_dir / "u_256_05000_burnin.npy", traj_dir)

    n_anim = 1
    trajs = load_trajectories_com(args.src_dir, every_nth_frame=100, max_trajs=n_anim)
    for idx in range(n_anim):
        animate_rlt_3d(trajs, idx=idx, fig_dir=traj_dirs[idx])

    plt_spectrum_burnin_3d(args.src_dir, max_trajs=args.max_trajs)
    plt_spectrum_points_3d(args.src_dir, max_trajs=args.max_trajs)
