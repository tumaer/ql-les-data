import os
from time import time

import jax
import jax.numpy as jnp
import numpy as np
from jax_sph.io_state import write_h5, write_vtk
from jax_sph.utils import pos_init_cartesian_3d

from l3es.turbulence import ur_to_u_dft_wrapper
from l3es.visualize import plot_e_k, plot_views


def get_ckps_list(files_root):
    files = os.listdir(files_root)
    files = [f for f in files if (".npy" in f)]
    files = sorted(files)
    return files


def shift_fn(r, dr, box_size=2 * np.pi):
    """Shift in a periodic box."""
    return (r + dr) % box_size


def spectral_interpolator_wrapper(N, splits=128):
    k = np.fft.fftshift(np.fft.fftfreq(N, 1.0 / N))
    k_field = np.array(np.meshgrid(k, k, k, indexing="xy"), dtype=int)  # (3, N, N, N)

    assert splits & (splits - 1) == 0, "Splits must be a power of 2"

    def interpolate(u, r):
        # u.shape = (3, N, N, N), r.shape - (N^3, 3)

        u_hat = jnp.fft.fftn(u, axes=(3, 2, 1))  # (3, N, N, N)
        u_hat = jnp.fft.fftshift(u_hat, axes=(1, 2, 3))

        def idft(carry, x_i):
            # 2 pi / L = 1
            exponent = jnp.exp(1j * (k_field * x_i[:, None, None, None]).sum(axis=0))
            res = jnp.real(jnp.mean(u_hat * exponent, axis=(1, 2, 3)))
            return None, res

        ### Version 1: Runtime on N=32 with vs without jit: 6s vs 35s
        # u_r = np.zeros_like(r)
        # for i, r_i in enumerate(r):
        #     u_r[i] = idft(None, r_i)[1]

        ### Version 2: Runtime on N=32: 1.35s
        # u_r = jax.lax.scan(idft, None, r)[1]

        ### Version 3: Runtime on N=32 and splits = 4...128: 0.23...0.26s linearly
        def body(carry, x_i):
            u_rs = jax.vmap(idft, in_axes=(None, 0))(None, x_i)[1]
            return None, u_rs

        r_splits = jnp.array(jnp.array_split(r, splits, axis=0))
        u_r = jax.lax.scan(body, None, r_splits)[1]
        u_r = jnp.concatenate(u_r, axis=0)

        return u_r

    return interpolate


# step = 1
# N = 32
# L = 2*np.pi
# data_path = "results_hit_192_3"
# vis_path = os.path.join(data_path, "int_vis")

# u_r = spectral_interpolator_wrapper(N, splits=8)(u, r)
# plot_views((r.T).reshape(3,N,N,N), (u_r.T).reshape(3,N,N,N), L/N, step, save_path=vis_path)


def integrate(src_path, dst_path, N=32, dt=0.01, splits=8):
    """Integrate SPH particles along prescribed velocity field.

    Args:
        src_path (str): Path to the directory with the checkpoints. (Source path)
        dst_path (str): Path to the directory where the integrated files will be saved.
        N (int): Number of particles in each dimension.
        dt (float): Time step.
        splits (int): Into how many parts to split 'r' before vmap-ing.
    """
    ckp_path = os.path.join(src_path, "ckp")
    files = get_ckps_list(ckp_path)

    vis_path = os.path.join(dst_path, "int_vis")
    int_path = os.path.join(dst_path, "int")
    os.makedirs(int_path, exist_ok=True)

    r = pos_init_cartesian_3d(2 * np.pi * np.ones(3), 2 * np.pi / N)

    L = 2 * np.pi
    dx = L / N

    interpolator = spectral_interpolator_wrapper(N, splits=8)
    t0 = time()
    t_int = 0.0
    for i, file in enumerate(files):
        u = np.load(os.path.join(ckp_path, file))

        t_temp = time()
        # r_eval = (r-dx/2) % L  # TODO: shift r by dx/2?
        u_r = interpolator(u, r)
        t_int += time() - t_temp

        #########################

        # import matplotlib.pyplot as plt

        # def plot_view(grid_x_s, us, labels = ["u", "u_irfft", "u_c",]):
        #     """Plot 3D views of scalar fields fields.
        #     Args:
        #         grid_x_s (list): List of grids (N_points, 3).
        #         us (list): List of scalar fields (N_points,).
        #     """
        #     def subplot_i(fig, ind, c, lbl, X):
        #         ax = fig.add_subplot(1, 2, ind, projection="3d")
        #         ax.view_init(elev=25.0, azim=-35, roll=0)
        #         ax.scatter(X[:,0], X[:,1], X[:,2], c=c, cmap="turbo", vmin=-4, vmax=4)
        #         ax.set_aspect("equal", "box")
        #         ax.set_title(f"{lbl} (min={c.min():.2f}, max={c.max():.2f})")
        #     # plot results
        #     fig = plt.figure(figsize=(10, 5))
        #     for i, (c, lbl, grid) in enumerate(zip(us, labels, grid_x_s)):
        #         subplot_i(fig, i + 1, c, lbl, grid)
        #     plt.savefig("test.png")

        # plot_view([r, r], [u.reshape(3,-1).T[:,0], u_r[:,0]], ["u_x", "u_r_x",])

        # # r_ = (r + dx/0.5 * np.random.rand(*r.shape)) % L
        # r_ = read_h5("results_hit_192_3/int/step_00100.h5")["r"]
        # u_r_ = interpolator(u, r_)
        # plot_view([r, r_], [u.reshape(3,-1).T[:,0], u_r_[:,0]], ["u_x", "u_r_x",])

        #########################

        write_h5({"r": r, "u": u_r}, os.path.join(int_path, f"step_{i:05d}.h5"))
        write_vtk({"r": r, "u": u_r}, os.path.join(int_path, f"step_{i:05d}.vtk"))

        if i % 100 == 0:
            r_vis = (r.T).reshape(3, N, N, N)
            u_vis = (u_r.T).reshape(3, N, N, N)
            plot_views(r_vis, u_vis, L / N, i, save_path=vis_path, u_ref=4.0)

            u_grid = ur_to_u_dft_wrapper(N, L)(r, u_r)  # TODO: shift r by dx/2?
            u_grid = np.asarray((u_grid.T).reshape(3, N, N, N))
            plot_e_k(u_grid, i, save_path=vis_path)

        r = shift_fn(r, dt * u_r)

    write_h5({"r": r, "u": u_r}, os.path.join(int_path, f"step_{i+1:05d}.h5"))
    write_vtk({"r": r, "u": u_r}, os.path.join(int_path, f"step_{i+1:05d}.vtk"))

    t_tot = time() - t0
    print(f"t_tot = {t_tot:.3f}, t_int = {t_int:.3f}")
