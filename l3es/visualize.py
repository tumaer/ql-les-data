import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jax import config

from l3es.utils import (
    comp_divergence,
    comp_divergence_3d,
    comp_vorticity,
    comp_vorticity_3d,
    energy_spectrum,
)

EPS = jnp.finfo(float).eps
config.update("jax_enable_x64", True)

# increase the font size in plots
plt.rcParams.update({"font.size": 14})


def plot_views(xyz, u, dx, step, field2=None, save_path=None, u_ref=4, suffix=""):
    """Scatter plot of the flow field

    Args:
    Type 1:
        xyz (np.ndarray): Grid of shape (3, N, N, N) or (3, N, N, 1).
        u (np.ndarray): Flow field with shape (3, N, N, N) or (3, N, N, 1).
    Type 2:
        xyz (np.ndarray): Grid of shape (3, N) or (2, N).
        u (np.ndarray): Flow field with shape (3, N) or (2, N).

        dx (float): Grid spacing.
        step (int): Current time step.
        field2 [key, value]: Additional field to plot.
        save_path (str): Results root directory.
        u_ref (float): Reference velocity for vmin and vmax in scatter plots.
    """
    x, y, z = xyz

    # x axis - bottom left, y axis - center, z axis - vertical
    assert xyz.shape == u.shape
    is_type = 2 if xyz.ndim == 2 else 1
    dim = 2 if (u.shape[-1] == 1 or u.shape[0] == 2) else 3
    Nx = len(x) if is_type == 1 else int(len(x) ** (1 / dim))
    if dim == 3:
        projection = "3d"
        mask = (x > 2 * np.pi - 1.4 * dx) | (y < 1.4 * dx) | (z > 2 * np.pi - 1.4 * dx)
        xyz_ = (x[mask], y[mask], z[mask])

        if field2 is None and is_type == 1:
            is_vorticity = True
            if is_vorticity:
                field2 = ["vort", comp_vorticity_3d(u, dx)[0]]
            else:
                field2 = ["div", comp_divergence_3d(u, dx, version=2)[0]]
            vmins = [-u_ref, None, 0]
            vmaxs = [u_ref, None, u_ref]
        elif field2 is None and is_type == 2:
            raise NotImplementedError("With type 2, field2 must be specified.")
        else:
            vmins = [-u_ref, 0.95, 0]
            vmaxs = [u_ref, 1.05, u_ref]

        fields = [u[0], field2[1], np.linalg.norm(u, axis=0)]
        labels = ["ux", field2[0], "|u|"]

    else:
        projection = None
        mask = np.ones_like(x[..., 0], dtype=bool)
        xyz_ = (x, y)
        u = u.squeeze()

        if field2 is None and is_type == 1:
            is_vorticity = True
            if is_vorticity:
                field2 = ["vort", comp_vorticity(u, dx)]
            else:
                field2 = ["div", comp_divergence(u, dx, version=2)]
            vmins = [-u_ref, None, 0]
            vmaxs = [u_ref, None, u_ref]
        elif field2 is None and is_type == 2:
            raise NotImplementedError("With type 2, field2 must be specified.")
        else:
            vmins = [-u_ref, 0.95, 0]
            vmaxs = [u_ref, 1.05, u_ref]

        labels = ["ux", field2[0], "|u|"]
        fields = [u[0], field2[1], np.linalg.norm(u, axis=0)]

    size = 36 * (32 / Nx) ** 2

    def subplot_i(ax, c, lbl, vmin, vmax):
        if c is None:
            return
        if dim == 3:
            ax.view_init(elev=25.0, azim=-35, roll=0)
        cmap = sns.color_palette("icefire", as_cmap=True) if lbl == "vort" else "turbo"
        ax.scatter(*xyz_, c=c[mask], cmap=cmap, s=size, vmin=vmin, vmax=vmax)
        ax.set_aspect("equal", "box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if dim == 3:
            ax.set_zlabel("z")
        ax.set_title(f"{lbl} (min={c.min():.2f}, max={c.max():.2f})")

    # plot results
    fig, axs = plt.subplots(
        1, 3, subplot_kw=dict(projection=projection), figsize=(15, 5)
    )
    for i, (ax, c, lbl, vn, vx) in enumerate(zip(axs, fields, labels, vmins, vmaxs)):
        subplot_i(ax, c, lbl, vn, vx)
    fig.tight_layout(pad=2)

    os.makedirs(save_path, exist_ok=True)
    if save_path:
        plt.savefig(os.path.join(save_path, f"step_{step}_view{suffix}.png"))
    else:
        plt.show()

    plt.close()


def plot_e_k(u, step, save_path=None, dim=3, ylims=(1e-4, 1e1), suffix=""):
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
    if dim == 3:
        ax.plot(k, k ** (-5 / 3), "--", c="k", label="k**(-5/3)")
    else:
        ax.plot(k, k ** (-2.0), "--", c="k", label="k**(-2)")

    ax.grid()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("E(k)")
    ax.legend()
    ax.set_ylim(ylims)
    fig.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f"step_{step}_spectrum{suffix}.png"))
    else:
        plt.show()

    plt.close()
