import csv
import os
from time import time

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax_sph.io_state import write_h5
from jax_sph.jax_md import space

from l3es.integrator import set_up_integrator, shift_fn
from l3es.spectral_solver import comp_dt, set_up_solver
from l3es.turbulence import u_and_spectrum_from_ur
from l3es.utils import spectral_filtering, write_u
from l3es.visualize import plot_e_k, plot_views


def combined(
    case,
    N=256,
    dim=3,
    nu=0.000625,
    t_final=0.1,
    t_burnin=0,
    dt=0.01,
    u_ref=4.0,
    ckp_N=64,
    # Interpolation
    state_0_path=None,
    interp_backend="dft",
    relax_dt_factor=0.0,
    dft_splits=8,
    # Global
    seed=42,
    rejit=True,
    # HIT forcing
    forcing_type="none",
    e_kin_target=None,
    kf=None,
    # IO
    log_freq=100,
    vis_freq=100,
    ckp_freq=100,
    dst_path=None,
):
    """Integrate SPH particles along prescribed velocity field.

    Args:
        src_path (str): Path to the directory with the checkpoints. (Source path)
        dst_path (str): Path to the directory where the integrated files will be saved.
        N (int): Number of particles in each dimension.
        dim (int): Dimension.
        t_burnin (float): Time to stop external forcing and start checkpointing.
        dt (float): Integration time step.
        dft_splits (int): Into how many parts to split 'r' before vmap-ing.
        relax_dt_factor (float): Factor applied to the relaxation time step.
        u_ref (float): Reference velocity for visualization and relaxation.
    """
    vis_path = os.path.join(dst_path, "com_vis")
    int_path = os.path.join(dst_path, "com")
    os.makedirs(int_path, exist_ok=True)
    diagnostics_path = os.path.join(dst_path, "diagnostics.csv")

    with open(diagnostics_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "time", "umax", "ekin", "dt_est", "e_inj", "rho_max"])

    u, u_hat, xyz_vis, dx_dns, integrate_fn, L, fft_axes, ek_ini, e_inj = set_up_solver(
        N, nu, dim, case, dt, seed, ckp_N, kf, e_kin_target, forcing_type, rejit
    )
    hit_eddy_turnover_time = 0.0
    print(
        f"{'#' * 79}\nSimulation with N={N}, nu={nu}, t_final={t_final}, dt={dt}",
        f"E_kin_init={ek_ini:.3f}\n{'#' * 79}",
    )

    r, comp_rho, interpolator, relax_fn, sph_fn, _, _ = set_up_integrator(
        state_0_path, dim, ckp_N, dft_splits, u_ref, interp_backend
    )
    displacement_fn, _ = space.periodic(side=L * np.ones(dim))
    displacement_fn_sets = vmap(displacement_fn)
    ugrid_from_ur = u_and_spectrum_from_ur(ckp_N, L, dim, fft_axes=fft_axes)

    t_sim, t_int, t0 = 0.0, 0.0, time()
    tstep_max = round(t_final / dt)
    burnin_steps = round(t_burnin / dt) if t_burnin > 0 else 0

    for i in range(tstep_max + 1):
        if i == burnin_steps:
            write_u(u, i, dst_path, N, suffix="_burnin")
        if i == burnin_steps and case == "HIT":
            # switch to a compiled solver without forcing
            _, _, _, _, integrate_fn, _, _, _, _ = set_up_solver(
                N, nu, dim, case, dt, seed, ckp_N, kf, e_kin_target, "none", rejit
            )
        t_temp = time()
        u, u_hat, e_inj = integrate_fn(u, u_hat)
        u.block_until_ready()
        t_sim += time() - t_temp

        # sum up injection rate to get eddy turnover time for forced HIT case
        hit_eddy_turnover_time += e_inj

        if i % vis_freq == 0:  # grid field
            if dim == 2:
                u_ckp = spectral_filtering(u[:2].squeeze(), ckp_N)[..., None]
            else:
                u_ckp = spectral_filtering(u, ckp_N)
            plot_views(
                xyz_vis,
                u_ckp,
                L / ckp_N,
                i,
                "vort",
                save_path=vis_path,
                u_ref=u_ref,
                suffix="_dns",
            )
            plot_e_k(u, i, save_path=vis_path, dim=dim, suffix="_dns")

        t_temp = time()
        u_hres = u if dim == 3 else u[:2, :, :, 0]  # (3,N,N,1) -> (2,N,N) if 2D
        assert ckp_N <= u.shape[1], "ckp_N must be less than or equal to N."
        assert (
            ckp_N % 2 == 0 and u.shape[1] % 2 == 0
        ), "Only tested for even N and ckp_N."
        u_lres = u_hres if ckp_N == u.shape[1] else spectral_filtering(u_hres, ckp_N)
        # u_lres.shape = (2, ckp_N, ckp_N) or (3, ckp_N, ckp_N, ckp_N)

        u_r = interpolator(u_lres, r)
        u_r_0 = u_r.copy()
        if relax_dt_factor > 0.0:
            a_r = np.zeros_like(r)
            r_0 = r.copy()
            # Shift and relax
            r_temp = shift_fn(r_0, dt * u_r)  # emulate "next step" to relax there
            for _ in range(1):
                # print(f"{comp_rho(r).max():.3f}, ", end='')
                a_temp = relax_fn(r_temp)
                r_temp = shift_fn(r_temp, (relax_dt_factor * dt) ** 2 * a_temp)
                a_r += a_temp * relax_dt_factor**2
            u_r = displacement_fn_sets(r_temp, r_0) / dt
        t_int += time() - t_temp

        rho, rho_max = None, None
        is_ckp = (i % ckp_freq == 0) and (i >= burnin_steps)
        if i % log_freq == 0 or is_ckp or i % vis_freq == 0:
            rho = comp_rho(r)
            rho_max = float(rho.max())

        if i % log_freq == 0:
            e_kin = 0.5 * float(jnp.mean(jnp.sum(u * u, axis=0)))
            umax = float(abs(u).max())
            dt_est = float(comp_dt(u, dx_dns, nu))
            sim_time = float(i * dt)
            with open(diagnostics_path, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [i, sim_time, umax, e_kin, dt_est, float(e_inj), rho_max]
                )
            print(
                f"Step {i}/{tstep_max}, u_max = {umax:.3f}, "
                f"E_kin = {e_kin:.3f}, dt_est = {dt_est:.5f}, "
                f"E_inj = {float(e_inj):.3f}, rho_max = {rho_max:.3f}"
            )

        if is_ckp:
            write_h5({"r": r, "u": u_r_0}, os.path.join(int_path, f"step_{i:05d}.h5"))
            print(f"Step {i}/{tstep_max}, rho_max={rho_max:.3f}.")

        if i % vis_freq == 0:  # particle field
            # Reshape particles to the shape of the grid field to reuse plt_views.
            # The reason to stick to grid-shaped fields in the optional vorticity or
            # divergence computation wtihin plot_views.
            r_vis = (r.T - 0.5 * L / ckp_N).reshape(*u_lres.shape)  # (3,N,N,N)|(2,N,N)
            u_vis = (u_r_0.T).reshape(*u_lres.shape)  # (3,N,N,N)|(2,N,N)
            if dim == 2:  # (2,N,N) -> (3,N,N,1)
                r_vis = np.vstack([r_vis, np.zeros_like(r_vis[:1])])[:, :, :, None]
                u_vis = np.vstack([u_vis, np.zeros_like(u_vis[:1])])[:, :, :, None]
            rho_vis = rho.reshape(*u_lres.shape[1:])  # (Ntot,) -> (N,N,N)|(N,N)
            print("Plotting to ", vis_path)
            plot_views(
                r_vis,
                u_vis,
                L / ckp_N,
                i,
                ["rho", rho_vis],
                save_path=vis_path,
                u_ref=u_ref,
                suffix="_sph",
            )
            # Plot energy spectrum
            u_grid = np.asarray(ugrid_from_ur(r, u_r_0)[0]).reshape(*u_lres.shape)
            if dim == 2:  # (2,N,N) -> (3,N,N,1)
                u_grid = np.vstack([u_grid, np.zeros_like(u_grid[:1])])[:, :, :, None]
            plot_e_k(u_grid, i, save_path=vis_path, dim=dim, suffix="_sph")
            print(f"Step {i}/{tstep_max} done.")

        r = shift_fn(r, dt * u_r)

    if case == "HIT" and forcing_type != "none":
        avg_e_inj = hit_eddy_turnover_time / tstep_max
        hit_eddy_turnover_time = avg_e_inj * kf**2
        hit_eddy_turnover_time = 1.0 / hit_eddy_turnover_time ** (1 / 3)
        print(f"Eddy turnover time =            {hit_eddy_turnover_time:.3f}")
        print(f"Number of eddy turnover times = {t_final / hit_eddy_turnover_time:.3f}")
        print(f"Average energy injected =       {avg_e_inj:.3f}")

        kmax = jnp.sqrt(2) * N / 3
        kolm_scale = (nu**3 / avg_e_inj) ** (0.25)
        print(f"eta=                            {kolm_scale:.3f}")
        print(f"kmax*eta =                      {kmax*kolm_scale:.3f}")

        e_kin = 0.5 * jnp.mean(jnp.sum(u * u, axis=0))
        Re_lambda = jnp.sqrt(20 * e_kin**2 / (3 * nu * avg_e_inj))
        print(f"Re_lambda =                     {Re_lambda:.3f}")

    t_tot = time() - t0
    print(f"t_tot = {t_tot:.3f}, t_sim = {t_sim}, t_int = {t_int:.3f}")
