import os
from time import time

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax_sph.io_state import read_h5, write_h5, write_vtk
from jax_sph.utils import pos_init_cartesian_2d, pos_init_cartesian_3d

from l3es.init_fields import init_u_hit, init_u_kolm, init_u_tgv, init_u_tgv2d
from l3es.integrator import shift_fn, spectral_interpolator_wrapper
from l3es.relax import relax_wrapper
from l3es.spectral_solver import comp_dt, rhs_wrapper, rk4_wrapper
from l3es.turbulence import ur_to_u_dft_wrapper, ur_to_u_mls_wrapper
from l3es.utils import rho_computer, spectral_filtering
from l3es.visualize import plot_e_k, plot_views


def combined(
    case,
    dst_path,
    state_0_path=None,
    N=256,
    ckp_N=64,
    dim=3,
    nu=0.000625,
    t_final=0.1,
    dt=0.01,
    splits=8,
    u_ref=4.0,
    relax=False,
    log_freq=100,
    vis_freq=100,
    ckp_freq=100,
    seed=42,
    debug=False,
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
    vis_path = os.path.join(dst_path, "com_vis")
    int_path = os.path.join(dst_path, "com")
    os.makedirs(int_path, exist_ok=True)

    L = 2 * np.pi
    dx_dns = L / N
    len_z = N if dim == 3 else 1
    fft_axes = (1, 2, 3) if dim == 3 else (1, 2)

    # set up spectral solver
    xyz = jnp.meshgrid(jnp.arange(N), jnp.arange(N), jnp.arange(len_z), indexing="ij")
    xyz = jnp.array(xyz) * L / N
    xyz_vis = (xyz.T + jnp.array([0, 0, L - dx_dns])).T if dim == 2 else xyz
    if case == "TGV":
        u = init_u_tgv(xyz[0], xyz[1], xyz[2])  # (3,N,N,N)
    elif case == "HIT":
        u = init_u_hit(N, seed)
    elif case == "Kolm":
        u = init_u_kolm(N, target_dim=3)
    elif case == "TGV2D":
        u = init_u_tgv2d(N, target_dim=3, rescale=L)
    u_hat = jnp.fft.rfftn(u, axes=fft_axes).squeeze()  # (3,N,N,N//2+1)
    t0 = time()
    integrate_fn = rk4_wrapper(dt, rhs_wrapper(N, nu, fft_axes), fft_axes)
    integrate_fn = jit(integrate_fn)
    u, u_hat = integrate_fn(u, u_hat)
    u.block_until_ready()
    print("Compilation time:", time() - t0)
    print("#" * 79, f"\nSimulation with N={N}, nu={nu}, t_final={t_final}, dt={dt}")
    t = 0.0
    t0 = time()
    t_sim = 0.0
    tstep_max = round(t_final / dt)

    # set up SPH particles
    accs = []
    if state_0_path is not None:
        r = read_h5(state_0_path)["r"]
    else:
        if dim == 3:
            r = pos_init_cartesian_3d(L * np.ones(3), L / ckp_N)
        else:
            r = pos_init_cartesian_2d(L * np.ones(2), L / ckp_N)
    comp_rho = rho_computer(ckp_N, dim=dim, L=L)
    interpolator = spectral_interpolator_wrapper(ckp_N, fft_axes, splits=splits)
    relax_fn = relax_wrapper(ckp_N, dim, L, is_physical=True, u_ref=u_ref)
    t_int = 0.0

    for i in range(tstep_max):
        t += dt
        t_temp = time()
        u, u_hat = integrate_fn(u, u_hat)
        u.block_until_ready()
        t_sim += time() - t_temp

        if i % log_freq == 0:
            e_kin = jnp.mean(jnp.sum(u * u, axis=0))
            print(
                f"Step {i}/{tstep_max}, u_max = {abs(u).max():.3f}, "
                f"E_kin = {e_kin:.3f}, dt_est = {comp_dt(u, dx_dns, nu):.5f}, ",
                end="",
            )
        if i % vis_freq == 0:
            plot_views(
                xyz_vis, u, dx_dns, i, save_path=vis_path, u_ref=u_ref, suffix="_dns"
            )
            plot_e_k(
                u, i, save_path=vis_path, dim=dim, ylims=(1e-8, 1e2), suffix="_dns"
            )
        # if i % ckp_freq == 0:
        #     write_u(u, i, int_path, ckp_N)

        t_temp = time()
        # r_eval = (r-dx/2) % L  # TODO: shift r by dx/2?
        u_hres = u if dim == 3 else u[:2, :, :, 0]
        assert ckp_N <= u.shape[1], "ckp_N must be less than or equal to N."
        assert (
            ckp_N % 2 == 0 and u.shape[1] % 2 == 0
        ), "Only tested for even N and ckp_N."
        u_lres = u_hres if ckp_N == u.shape[1] else spectral_filtering(u_hres, ckp_N)
        # # evaluate the spectrum on the input field as a sanity check -> looks fine!
        # if dim == 2:
        #     u_input = np.vstack([u_lres, np.zeros_like(u_lres[:1])])[:, :, :, None]
        # plot_e_k(
        #     u_input, i, save_path=vis_path, dim=dim, ylims=(1e-8,1e2), suffix="_val"
        # )

        u_r = interpolator(u_lres, r)
        u_r_0 = u_r.copy()
        if relax:
            a_r = np.zeros_like(r)
            if debug:
                a_r_s = []
            if debug:
                print(f"Relax ({i},0): {u_r.max():.4f} [", end="")
            r_temp = shift_fn(r, dt * u_r)  # emulate "next step" to relax there
            for _ in range(10):
                # print(f"{comp_rho(r).max():.3f}, ", end='')
                a_temp = relax_fn(r_temp)
                r_temp = shift_fn(r_temp, dt**2 * a_temp)
                if debug:
                    print(f"{a_temp.max():.3f}, ", end="")

                if debug:
                    a_r_s.append(a_temp * dt)
                a_r += a_temp
            if debug:
                print(f"] Relax ({i},1): {u_r.max():.4f}, {a_r.max():.4f} ")

            a_max = a_r.max()
            accs.append(a_max)
            if a_max > 100_000:
                print("Breaking.")
                break
            u_r += dt * a_r
        if debug:
            u_r.block_until_ready()
        t_int += time() - t_temp

        if i % ckp_freq == 0:
            if not debug:
                write_h5(
                    {"r": r, "u": u_r_0}, os.path.join(int_path, f"step_{i:05d}.h5")
                )
                print(f"Step {i}/{tstep_max}, rho_max={comp_rho(r).max():.3f}.")
            else:
                out_dict = {
                    "r": r,
                    "u_r": u_r,
                    "u_r_0": u_r_0,
                    "a_r_dt": a_r * dt,
                    "rho": comp_rho(r),
                }
                for ii, temprary_a in enumerate(a_r_s):
                    out_dict[f"a_r_{ii}_dt"] = temprary_a
                write_vtk(out_dict, os.path.join(int_path, f"step_{i:05d}.vtk"))
                print(f"Step {i}/{tstep_max}, rho_max={out_dict['rho'].max():.3f}.")

        if i % vis_freq == 0:
            r_vis = (r.T).reshape(*u_lres.shape)
            u_vis = (u_r_0.T).reshape(*u_lres.shape)
            if dim == 2:
                r_vis = np.vstack([r_vis, np.zeros_like(r_vis[:1])])[:, :, :, None]
                u_vis = np.vstack([u_vis, np.zeros_like(u_vis[:1])])[:, :, :, None]
            rho = comp_rho(r).reshape(*u_lres.shape[1:])
            print("Plotting to ", vis_path)
            plot_views(
                r_vis,
                u_vis,
                L / ckp_N,
                i,
                rho,
                save_path=vis_path,
                u_ref=u_ref,
                suffix="_sph",
            )
            # TODO: shift r by dx/2?
            is_dft_to_grid = False  # MLS works better!
            if is_dft_to_grid:
                u_grid = ur_to_u_dft_wrapper(ckp_N, L, dim, fft_axes)(r, u_r_0)
            else:
                u_grid = ur_to_u_mls_wrapper(ckp_N, L, dim)(r, u_r_0)
            u_grid = np.asarray((u_grid.T).reshape(*u_lres.shape))
            if dim == 2:
                u_grid = np.vstack([u_grid, np.zeros_like(u_grid[:1])])[:, :, :, None]
            plot_e_k(
                u_grid, i, save_path=vis_path, dim=dim, ylims=(1e-8, 1e2), suffix="_sph"
            )
            print(f"Step {i}/{tstep_max} done.")

        r = shift_fn(r, dt * u_r)

    # write_h5({"r": r, "u": u_r}, os.path.join(int_path, f"step_{i+1:05d}.h5"))
    write_vtk({"r": r, "u": u_r_0}, os.path.join(int_path, f"step_{i+1:05d}.vtk"))

    t_tot = time() - t0
    print(f"t_tot = {t_tot:.3f}, t_sim = {t_sim}, t_int = {t_int:.3f}")

    # import pickle
    # with open(os.path.join(dst_path, "accs.pkl"), "wb") as f:
    #     pickle.dump(all_accs, f)
