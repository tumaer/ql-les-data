import os
from time import time

import jax.numpy as jnp
import numpy as np
from jax import tree_map, vmap
from jax_sph.io_state import write_h5, write_vtk
from jax_sph.jax_md import space

from l3es.integrator import set_up_integrator, shift_fn
from l3es.spectral_solver import comp_dt, set_up_solver
from l3es.turbulence import ur_to_u_dft_wrapper, ur_to_u_mls_wrapper
from l3es.utils import (
    make_incompressible_real,
    make_incompressible_spectral,
    spectral_filtering,
)
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

    u, u_hat, xyz_vis, dx_dns, integrate_fn, L, fft_axes = set_up_solver(
        N, nu, dim, case, dt, seed
    )
    print("#" * 79, f"\nSimulation with N={N}, nu={nu}, t_final={t_final}, dt={dt}")
    t = 0.0
    t_sim = 0.0
    tstep_max = round(t_final / dt) + 1

    r, comp_rho, interpolator, relax_fn, sph_fn, _, _ = set_up_integrator(
        state_0_path, dim, ckp_N, splits, u_ref
    )
    displacement_fn, _ = space.periodic(side=L * np.ones(dim))
    displacement_fn_sets = vmap(displacement_fn)
    # accs = []
    # u_r_old = jnp.zeros_like(r)
    t_int = 0.0
    t0 = time()

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
                end="" if log_freq >= ckp_freq else "\n",
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
        # u_lres.shape = (2, ckp_N, ckp_N)
        # # evaluate the spectrum on the input field as a sanity check -> looks fine!
        # if dim == 2:
        #     u_input = np.vstack([u_lres, np.zeros_like(u_lres[:1])])[:, :, :, None]
        # plot_e_k(
        #     u_input, i, save_path=vis_path, dim=dim, ylims=(1e-8,1e2), suffix="_val"
        # )

        incompr_type = None  # "real", "spectral", or None
        # Overall, incompressility is not the reason for having to relax particles.
        if incompr_type == "spectral":
            # Make vel. field incompressible in spectral space, i.e. ik_i u_i = 0
            # Nothing will change in the field as filtering doesn't affect it.
            u_temp = np.vstack([u_lres, np.zeros_like(u_lres[:1])])[:, :, :, None]
            u_lres, _ = make_incompressible_spectral(ckp_N, fft_axes)(u=u_temp)
            # u_lres.shape = (3, ckp_N, ckp_N)
        elif incompr_type == "real":
            # Make vel. field incompressible when evaluated with finite difference
            u_lres = make_incompressible_real(u_lres, ckp_N, L).squeeze()
            # u_lres.shape = (3, ckp_N, ckp_N)
        if incompr_type is not None and dim == 2:
            u_lres = u_lres[:2]  # (3, ckp_N, ckp_N) -> (2, ckp_N, ckp_N)

        # import matplotlib.pyplot as plt
        # from l3es.utils import comp_divergence
        # fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        # def plt_field(ax, v, dx=None, mode=None):
        #     # v.shape=(2,N,N) if mode="div" else (N,N)
        #     vext=5
        #     if mode == "div":
        #         v = comp_divergence(v, dx)
        #         vext=1.5
        #     ax.imshow(v, vmin=-vext, vmax=vext)
        #     label = "div(u)" if mode == "div" else "u"
        #     ax.set_title(
        #         f"{label} min/max/std: [{v.min():.2f}, {v.max():.2f}, {v.std():.2f}]"
        #     )
        # u_lres_inc is the u_lres after making it incompressible
        # plt_field(axs[0,0], u_hres[0,:,:])
        # plt_field(axs[0,1], u_lres[0,:,:])
        # plt_field(axs[0,2], u_lres_inc[0,:,:])
        # plt_field(axs[1,0], u_hres[:,:,:], dx=dx_dns, mode="div")
        # plt_field(axs[1,1], u_lres[:,:,:], dx=L/ckp_N, mode="div")
        # plt_field(axs[1,2], u_lres_inc[:,:,:], dx=L/ckp_N, mode="div")
        # plt.tight_layout()
        # plt.savefig(f"incompr_figure.png")

        u_r = interpolator(u_lres, r)
        u_r_0 = u_r.copy()
        if relax:
            a_r = np.zeros_like(r)
            if debug:
                a_r_s = []
            if debug:
                print(f"Relax ({i},0): {u_r.max():.4f} [", end="")
            r_0 = r.copy()
            is_shift_and_relax = True
            if is_shift_and_relax:
                r_temp = shift_fn(r_0, dt * u_r)  # emulate "next step" to relax there
                dt_factor = 2  # if dt gives CFL=0.4, then 2*dt gives CFL=0.8
                for _ in range(1):
                    # print(f"{comp_rho(r).max():.3f}, ", end='')
                    a_temp = relax_fn(r_temp)
                    r_temp = shift_fn(r_temp, (dt_factor * dt) ** 2 * a_temp)
                    if debug:
                        print(f"{a_temp.max():.3f}, ", end="")

                    if debug:
                        a_r_s.append(a_temp * dt)
                    a_r += a_temp * dt_factor**2
                if debug:
                    print(f"] Relax ({i},1): {u_r.max():.4f}, {a_r.max():.4f} ")
            else:  # do an actual TVF-SPH step
                acc = sph_fn(r_0, u_r)
                if debug:
                    magnitudes = tree_map(
                        lambda x: str(jnp.linalg.norm(x).item())[:8], acc
                    )
                    print(f"] Relax: {magnitudes}")
                else:
                    if i % log_freq == 0:
                        magnitudes = tree_map(
                            lambda x: str(jnp.linalg.norm(x).item())[:8], acc
                        )
                        print(f"{magnitudes}, ", end="")
                dudt = acc["acc_p"] + nu * acc["acc_visc"]
                dvdt = acc["acc_tvf"]
                u_new = u_r + dt * dudt
                v_new = u_new + 0.5 * dt * dvdt
                r_temp = shift_fn(r_0, dt * v_new)

                dt_factor = 1
                for _ in range(3):
                    # print(f"{comp_rho(r).max():.3f}, ", end='')
                    a_temp = relax_fn(r_temp)
                    r_temp = shift_fn(r_temp, (dt_factor * dt) ** 2 * a_temp)

            # a_max = a_r.max()
            # accs.append(a_max)
            # if a_max > 100_000:
            #     print("Breaking.")
            #     break
            # because here we have a different dt, we infer vel from displacement
            # u_r += dt * a_r  # this should actually also work
            u_r = displacement_fn_sets(r_temp, r_0) / dt
            # print(
            #     "magn a: ",
            #     jnp.linalg.norm(u_r - u_r_0).item(),   # 20...30
            #     jnp.linalg.norm(u_r_0 - u_r_old).item()  # 0,7
            # )
            # u_r_old = u_r_0.copy()
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
                ["rho", rho],
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

    t_tot = time() - t0
    print(f"t_tot = {t_tot:.3f}, t_sim = {t_sim}, t_int = {t_int:.3f}")
