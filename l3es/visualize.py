import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import config

from l3es.utils import energy_spectrum

EPS = jnp.finfo(float).eps
config.update("jax_enable_x64", True)

# increase the font size in plots
plt.rcParams.update({"font.size": 14})


def plot_views(xyz, u, dx, step, save_path=None, u_ref=4):
    """Scatter plot of the flow field

    Args:
        xyz (np.ndarray): Grid of shape (3, N, N, N) or (3, N, N, 1).
        u (np.ndarray): Flow field with shape (3, N, N, N) or (3, N, N, 1).
        dx (float): Grid spacing.
        step (int): Current time step.
        save_path (str): Results root directory.
        u_ref (float): Reference velocity for vmin and vmax in scatter plots.
    """
    x, y, z = xyz

    # x axis - bottom left, y axis - center, z axis - vertical
    dim = 2 if u.shape[-1] == 1 else 3
    if dim == 3:
        projection = "3d"
        mask = (x > 2 * np.pi - 1.4 * dx) + (y < 1.4 * dx) + (z > 2 * np.pi - 1.4 * dx)
        xyz_ = (x[mask], y[mask], z[mask])

        fields = [u[0], u[1], u[2], np.linalg.norm(u, axis=0)]
        labels = ["ux", "uy", "uz", "|u|"]
        vmins = [-u_ref, -u_ref, -u_ref, 0]
    else:
        projection = None
        mask = np.ones_like(x[..., 0], dtype=bool)
        xyz_ = (x, y)
        u = u.squeeze()

        fields = [u[0], u[1], np.linalg.norm(u, axis=0)]
        labels = ["ux", "uy", "|u|"]
        vmins = [-u_ref, -u_ref, 0]

    size = 36 * (32 / x.shape[0]) ** 2

    def subplot_i(ax, ind, c, lbl, vmin):
        if dim == 3:
            ax.view_init(elev=25.0, azim=-35, roll=0)
        ax.scatter(*xyz_, c=c[mask], cmap="turbo", s=size, vmin=vmin, vmax=u_ref)
        ax.set_aspect("equal", "box")
        ax.set_title(f"{lbl} (min={c.min():.2f}, max={c.max():.2f})")

    # plot results
    fig, axs = plt.subplots(
        1, dim + 1, subplot_kw=dict(projection=projection), figsize=((dim + 1) * 5, 5)
    )
    for i, (ax, c, lbl, vmin_i) in enumerate(zip(axs, fields, labels, vmins)):
        subplot_i(ax, i + 1, c, lbl, vmin_i)
    fig.tight_layout(pad=2)

    os.makedirs(save_path, exist_ok=True)
    if save_path:
        plt.savefig(os.path.join(save_path, f"step_{step}_view.png"))

    plt.close()


def plot_e_k(u, step, save_path=None, dim=3):
    """Plot energy spectrum.

    Args:
        u (np.ndarray): Flow field with shape (3, N, N, N) or (3, N, N, 1).
        step (int): Current time step.
        save_path (str): Where to write results.
        dim (int): Dimension of the flow field.
    """
    n = u.shape[1]
    ek = np.asarray(energy_spectrum(u, dim=dim))
    k = np.arange(1, len(ek))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.axvline(n // 2, c="tab:orange", ls="--", label="N/2")
    ax.axvline(n // 3, c="tab:green", ls="--", label="N/3")
    ax.plot(k, ek[1:])
    ax.set_title("E(k) with k=(0, n]")
    ax.plot(k, k ** (-5 / 3), "--", c="k", label="k**(-5/3)")

    ax.grid()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("E(k)")
    ax.legend()
    ax.set_ylim(1e-4, 1e1)
    fig.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    if save_path:
        fig.savefig(os.path.join(save_path, f"step_{step}_spectrum.png"))

    plt.close()
