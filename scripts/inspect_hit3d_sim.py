"""Visualizations for data generated with:
`python main.py config=configs/hit32_1.yaml mode=simulate sim.ckp_freq=100`.
"""
import argparse
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from inspect_hit3d_com import plot_visible_planes_3d
from matplotlib.colors import Normalize
from utils import _ls_sorted_frames, _step_from_filename, plt_diagnostics_sim

from l3es.utils import energy_spectrum


def plt_spectrum_over_time(ds_root: Path, n_ckp=5) -> None:
    """Plot energy spectrum for the first checkpoints in a simulation."""
    ckp_files = _ls_sorted_frames(ds_root, mode="sim", match="u_*.npy")
    ckp_files = ckp_files[:: (len(ckp_files) // n_ckp)]

    fig, ax = plt.subplots(figsize=(5.5, 5))

    k = None
    for file in ckp_files:
        u = np.load(file)
        spectrum = energy_spectrum(u, dim=3)
        k = np.arange(1, len(spectrum))
        ax.plot(k, spectrum[1:], label="step=" + str(_step_from_filename(file)))

    if k is None:
        raise RuntimeError(f"No checkpoint files found in {ds_root / 'sim'}.")

    n = len(k)
    ax.axvline(n // 2, c="tab:orange", ls="--", label="N/2")
    ax.axvline(n // 3, c="tab:green", ls="--", label="N/3")
    ax.plot(k, k ** (-5 / 3), "--", c="k", label="k^(-5/3)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("E(k)")
    ax.set_ylim(1e-8, None)
    ax.grid()
    ax.set_title("Energy Spectrum over Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ds_root / "energy_spectrum_over_time.png", dpi=200)
    plt.close()
    print("Finished plt_spectrum_over_time !")


def animate_field_over_time(ds_root: Path) -> None:
    """Animate |u| as 3D visible side planes across checkpoints."""
    files = _ls_sorted_frames(ds_root, mode="sim", match="u_*.npy")
    fields = []
    steps = []
    for file in files:
        u = np.load(file)
        u_norm = np.linalg.norm(u, axis=0)
        fields.append(u_norm)
        steps.append(_step_from_filename(file))

    if not fields:
        raise RuntimeError(f"No checkpoint files found in {ds_root / 'sim'}.")

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    norm = Normalize(vmin=0, vmax=4)
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap="turbo")
    scalar_mappable.set_array([])
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(scalar_mappable, cax=cax)

    length = len(fields)
    plot_visible_planes_3d(
        ax,
        fields[0],
        vmin=0,
        vmax=4,
        title=f"|u| at step={steps[0]}",
        elev=25.0,
        azim=-35,
        roll=0,
    )

    def update(frame):
        ax.cla()
        plot_visible_planes_3d(
            ax,
            fields[frame],
            vmin=0,
            vmax=4,
            title=f"|u| at step={steps[frame]}",
            elev=25.0,
            azim=-35,
            roll=0,
        )
        return tuple(ax.collections)

    anim = animation.FuncAnimation(fig, update, frames=length, interval=100, blit=False)
    anim.save(ds_root / "anim.gif", writer="pillow", fps=10)
    plt.close()
    print("Finished animate_field_over_time !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", default=Path("results/hit32_1"), type=Path)
    args = parser.parse_args()
    ds_root = args.src_dir
    plt_diagnostics_sim(ds_root)
    plt_spectrum_over_time(ds_root)
    animate_field_over_time(ds_root)
