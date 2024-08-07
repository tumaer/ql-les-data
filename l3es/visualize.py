import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import config

from l3es.utils import energy_spectrum

EPS = jnp.finfo(float).eps
config.update("jax_enable_x64", True)


def plot_views(xyz, u, dx, step, save_path=None, vmin=-4, vmax=4):
    x, y, z = xyz

    mask = (x > 2 * np.pi - 1.5 * dx) + (y < 0.5 * dx) + (z > 2 * np.pi - 1.5 * dx)

    def subplot_i(fig, ind, c, lbl, vmin):
        ax = fig.add_subplot(1, 4, ind, projection="3d")
        ax.view_init(elev=25.0, azim=-35, roll=0)
        cmp = "turbo"
        ax.scatter(x[mask], y[mask], z[mask], c=c[mask], cmap=cmp, vmin=vmin, vmax=vmax)
        ax.set_aspect("equal", "box")
        ax.set_title(lbl)

    # plot results
    fig = plt.figure(figsize=(20, 5))
    fields = [u[0], u[1], u[2], np.linalg.norm(u, axis=0)]
    labels = ["ux", "uy", "uz", "|u|"]
    vmins = [vmin, vmin, vmin, 0] if vmin is not None else [None] * 4
    for i, (c, lbl, vmin_i) in enumerate(zip(fields, labels, vmins)):
        subplot_i(fig, i + 1, c, lbl, vmin_i)

    os.makedirs(save_path, exist_ok=True)
    if save_path:
        plt.savefig(os.path.join(save_path, f"step_{step}_view.png"))

    plt.close()


def plot_e_k(u, step, save_path=None):
    N = u.shape[-1]
    ek = energy_spectrum(u)
    k = np.arange(1, len(ek))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(k, (ek[1:]))
    axs[0].plot(k, k ** (-5 / 3), "--", c="k", label="k**(-5/3)")
    axs[0].set_ylabel("E(k)")
    axs[0].set_xlabel("Wavenumber k")
    axs[0].legend()
    axs[0].grid()
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_ylim(1e-4, 1e0)
    axs[0].set_title("E(k) with k=(0, n]")

    axs[1].plot(k[: N // 2 + 1], (ek[1 : N // 2 + 2]))
    axs[1].plot(
        k[: N // 2 + 1], k[: N // 2 + 1] ** (-5 / 3), "--", c="k", label="k**(-5/3)"
    )
    axs[1].set_ylabel("E(k)")
    axs[1].set_xlabel("Wavenumber k")
    axs[1].legend()
    axs[1].grid()
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_ylim(1e-4, 1e0)
    axs[1].set_title("E(k) with k=(0, n/2 + 1]")
    fig.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    if save_path:
        fig.savefig(os.path.join(save_path, f"step_{step}_spectrum.png"))

    plt.close()
