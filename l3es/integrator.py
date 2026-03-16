import os
from time import time

import finufft
import jax
import jax.numpy as jnp
import numpy as np
from jax_sph.io_state import read_h5, write_h5, write_vtk
from jax_sph.utils import pos_init_cartesian_2d, pos_init_cartesian_3d

from l3es.relax import relax_wrapper
from l3es.turbulence import ur_to_u_dft_wrapper, ur_to_u_mls_wrapper
from l3es.utils import rho_computer
from l3es.visualize import plot_e_k, plot_views


def get_ckps_list(files_root):
    files = os.listdir(files_root)
    files = [f for f in files if (".npy" in f)]
    files = sorted(files)
    return files


def shift_fn(r, dr, box_size=2 * np.pi):
    """Shift in a periodic box."""
    return (r + dr) % box_size


def spectral_interpolator_wrapper(N, fft_axes=(1, 2, 3), splits=128, backend="nufft"):
    is_3d = len(fft_axes) == 3
    k = np.fft.fftshift(np.fft.fftfreq(N, 1.0 / N))
    k_tuple = (k, k, k) if is_3d else (k, k)
    k_field = np.array(np.meshgrid(*k_tuple, indexing="ij"), dtype=int)  # (3, N, N, N)
    norm = N ** len(fft_axes)

    assert splits & (splits - 1) == 0, "Splits must be a power of 2"
    assert backend in ["dft", "nufft"]

    def interpolate_finufft(u, r):
        # u.shape = (3, N, N, N) or (2, N, N), r.shape = (num_particles, dim)
        u_hat = np.fft.fftshift(
            np.fft.fftn(np.asarray(u), axes=fft_axes), axes=fft_axes
        )

        # 1. Make the entire batched coefficient array contiguous ONCE
        # FINUFFT expects shape (n_trans, N, N, N) for multi-transforms
        coeff = np.ascontiguousarray(u_hat, dtype=np.complex128)

        # Extract and format coordinates
        x = np.ascontiguousarray(r[:, 0], dtype=np.float64)
        y = np.ascontiguousarray(r[:, 1], dtype=np.float64)
        if is_3d:
            z = np.ascontiguousarray(r[:, 2], dtype=np.float64)
            # 2. Vectorized call: passing the (3, N, N, N) coeff array directly
            # Returns vals of shape (3, num_particles)
            vals = finufft.nufft3d2(x, y, z, coeff, isign=1)
        else:
            vals = finufft.nufft2d2(x, y, coeff, isign=1)

        # 3. Transpose the result to match (num_particles, channels) and normalize
        u_r = np.real(vals).T / norm

        return jnp.asarray(u_r)

    def interpolate_dft(u, r):
        # u.shape = (3, N, N, N) or (2, N, N), r.shape - (N^3, 3)

        u_hat = jnp.fft.fftn(u, axes=fft_axes)  # (3, N, N, N)
        u_hat = jnp.fft.fftshift(u_hat, axes=fft_axes)

        def idft(carry, x_i):
            # 2 pi / L = 1
            exponent = jnp.exp(1j * ((k_field.T * x_i).T).sum(axis=0))
            res = jnp.real(jnp.mean(u_hat * exponent, axis=fft_axes))
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

    return interpolate_dft if backend == "dft" else interpolate_finufft


# step = 1
# N = 32
# L = 2*np.pi
# data_path = "results_hit_192_3"
# vis_path = os.path.join(data_path, "int_vis")

# u_r = spectral_interpolator_wrapper(N, splits=8)(u, r)
# plot_views(
#     (r.T).reshape(3,N,N,N), (u_r.T).reshape(3,N,N,N), L/N, step, save_path=vis_path
# )


def my_imshow(ax, u, vmin, vmax):
    ax.imshow(u.T, origin="lower", cmap="turbo", vmin=vmin, vmax=vmax)
    ax.set_title(f"ux (min={u.min():.2f}, max={u.max():.2f})")


def set_up_integrator(state_0_path, dim, N, splits, u_ref, interp_backend):
    L = 2 * np.pi
    fft_axes = (1, 2, 3) if dim == 3 else (1, 2)

    if state_0_path is not None:
        r = read_h5(state_0_path)["r"]
    else:
        if dim == 3:
            r = pos_init_cartesian_3d(L * np.ones(3), L / N)
        else:
            r = pos_init_cartesian_2d(L * np.ones(2), L / N)

    comp_rho = rho_computer(N, dim=dim, L=L)
    interpolator = spectral_interpolator_wrapper(
        N, fft_axes, splits=splits, backend=interp_backend
    )
    relax_fn, sph_fn = relax_wrapper(N, dim, L, is_physical=True, u_ref=u_ref)

    return r, comp_rho, interpolator, relax_fn, sph_fn, L, fft_axes


def integrate(
    src_path,
    dst_path,
    state_0_path=None,
    N=32,
    dim=3,
    dt=0.01,
    splits=8,
    u_ref=4.0,
    relax=False,
    vis_freq=100,
    interp_backend="dft",
):
    """Integrate SPH particles along prescribed velocity field.

    Args:
        src_path (str): Path to the directory with the checkpoints. (Source path)
        dst_path (str): Path to the directory where the integrated files will be saved.
        N (int): Number of particles in each dimension.
        dim (int): Dimension.
        dt (float): Integration time step.
        splits (int): Into how many parts to split 'r' before vmap-ing.
        relax (bool): whether to relax the coordinates.
        u_ref (float): Reference velocity for visualization and relaxation.
    """

    ckp_path = os.path.join(src_path, "ckp")
    files = get_ckps_list(ckp_path)

    vis_path = os.path.join(dst_path, "int_vis")
    int_path = os.path.join(dst_path, "int")
    os.makedirs(int_path, exist_ok=True)

    r, comp_rho, interpolator, relax_fn, _, L, fft_axes = set_up_integrator(
        state_0_path, dim, N, splits, u_ref, interp_backend
    )
    all_accs = {}
    # for factor in [5, 7, 10, 15, 20, 25]:
    factor = 20
    accs = []
    t0 = time()
    t_int = 0.0
    # r = shift_fn(r, jax.random.normal(jax.random.key(42), r.shape) * 0.1 * L / N)
    for i, file in enumerate(files):
        u = np.load(os.path.join(ckp_path, file))
        # evaluate the spectrum on the input field as a sanity check -> looks fine!
        # if dim == 2:
        #     u_input = np.vstack([u, np.zeros_like(u[:1])])[:, :, :, None]
        # plot_e_k(u_input, i, save_path=vis_path, dim=dim)

        t_temp = time()
        # r_eval = (r-dx/2) % L  # TODO: shift r by dx/2?
        u_r = interpolator(u, r)
        u_r_0 = u_r.copy()
        if relax:
            # r_temp = shift_fn(r, dt * u_r)  # switching order doesn't change anything
            # r_temp = r
            # a_r = factor * relax_fn(r_temp)
            a_r = np.zeros_like(r)
            a_r_s = []
            print(f"Relax ({i},0): {u_r.max():.4f} [", end="")
            r_temp = shift_fn(r, dt * u_r)  # emulate "next step" to relax there
            for _ in range(10):
                # print(f"{comp_rho(r).max():.3f}, ", end='')
                a_temp = relax_fn(r_temp)
                r_temp = shift_fn(r_temp, dt**2 * a_temp)
                print(f"{a_temp.max():.3f}, ", end="")

                a_r_s.append(a_temp * dt)
                a_r += a_temp
            print(f"] Relax ({i},1): {u_r.max():.4f}, {a_r.max():.4f} ", end="")

            a_max = a_r.max()
            accs.append(a_max)
            if a_max > 100_000:
                print("Breaking.")
                break
            u_r += dt * a_r
        t_int += time() - t_temp

        # import matplotlib.pyplot as plt
        # _, axs = plt.subplots(1, 2, figsize=(10, 5))
        # my_imshow(axs[0], u[0, :, :], -u_ref, u_ref)
        # axs[1].scatter(r[:,0], r[:,1], s=36 * (32 / N) ** 2, c=u_r[:,0],
        #                cmap="turbo", vmin=-u_ref, vmax=u_ref)
        # axs[1].set_title(
        #     f"u_r_x (min={u_r[:,0].min():.2f}, max={u_r[:,0].max():.2f})")
        # plt.savefig("orientation_check.png")

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

        # write_h5({"r": r, "u": u_r}, os.path.join(int_path, f"step_{i:05d}.h5"))
        out_dict = {
            "r": r,
            "u_r": u_r_0,
            "u_r_0": u_r_0,
            "a_r_dt": a_r * dt,
            "rho": comp_rho(r),
        }
        for ii, temprary_a in enumerate(a_r_s):
            out_dict[f"a_r_{ii}_dt"] = temprary_a
        write_vtk(out_dict, os.path.join(int_path, f"step_{i:05d}.vtk"))
        print(f"Step {i}/{len(files)}, rho_max={out_dict['rho'].max():.3f}.")

        if i % vis_freq == 0:
            r_vis = (r.T).reshape(*u.shape)
            u_vis = (u_r.T).reshape(*u.shape)
            if dim == 2:
                r_vis = np.vstack([r_vis, np.zeros_like(r_vis[:1])])[:, :, :, None]
                u_vis = np.vstack([u_vis, np.zeros_like(u_vis[:1])])[:, :, :, None]
            rho = comp_rho(r).reshape(*u.shape[1:])
            print("plotting to ", vis_path)
            field2 = ["rho", rho]
            plot_views(r_vis, u_vis, L / N, i, field2, save_path=vis_path, u_ref=u_ref)
            # TODO: shift r by dx/2?
            is_dft_to_grid = False  # MLS works better!
            if is_dft_to_grid:
                u_grid = ur_to_u_dft_wrapper(N, L, dim, fft_axes)(r, u_r_0)
            else:
                u_grid = ur_to_u_mls_wrapper(N, L, dim)(r, u_r_0)
            u_grid = np.asarray((u_grid.T).reshape(*u.shape))
            if dim == 2:
                u_grid = np.vstack([u_grid, np.zeros_like(u_grid[:1])])[:, :, :, None]
            plot_e_k(u_grid, i, save_path=vis_path, dim=dim)
            print(f"Step {i}/{len(files)} done.")

        r = shift_fn(r, dt * u_r)

    write_h5({"r": r, "u": u_r}, os.path.join(int_path, f"step_{i+1:05d}.h5"))
    write_vtk({"r": r, "u": u_r}, os.path.join(int_path, f"step_{i+1:05d}.vtk"))

    t_tot = time() - t0
    print(f"t_tot = {t_tot:.3f}, t_int = {t_int:.3f}")
    all_accs[factor] = np.array(accs)

    # import pickle
    # with open(os.path.join(dst_path, "accs.pkl"), "wb") as f:
    #     pickle.dump(all_accs, f)
