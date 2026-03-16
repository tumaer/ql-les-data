"""Core spectral DNS solver."""

import csv
import os
import pickle
from time import time

import jax
import jax.numpy as jnp
from jax import config
from jax.experimental import serialize_executable

from l3es.init_fields import (
    init_forcing_mask_hit,
    init_u_hit,
    init_u_kolm,
    init_u_tgv,
    init_u_tgv2d,
)
from l3es.utils import spectral_filtering, write_u
from l3es.visualize import plot_e_k, plot_views

EPS = jnp.finfo(float).eps
config.update("jax_enable_x64", True)


def rhs_wrapper(N, nu, fft_axes=(1, 2, 3)):
    """Spectral Navier-Stokes solver right hand side.

    Based on: https://github.com/spectralDNS/spectralDNS
    """
    kx = jnp.fft.fftfreq(N, 1.0 / N)
    kx_tuple = (kx,) * 2 if len(fft_axes) == 3 else (kx,)
    kz = kx[: (N // 2 + 1)].copy()
    # kz[-1] *= -1
    kz = kz.at[-1].set(-1 * kz[-1])
    kkk = jnp.array(jnp.meshgrid(*kx_tuple, kz, indexing="ij"), dtype=int)
    if len(fft_axes) == 2:
        kkk = jnp.concatenate([kkk, jnp.zeros((1, N, N // 2 + 1))], axis=0)
    kkk2 = jnp.sum(kkk * kkk, 0, dtype=int)
    kkk_over_kkk2 = kkk.astype(float) / jnp.where(kkk2 == 0, 1, kkk2).astype(float)
    kmax_dealias = 2.0 / 3.0 * (N // 2 + 1)
    dealias = jnp.array(
        (abs(kkk[0]) < kmax_dealias)
        * (abs(kkk[1]) < kmax_dealias)
        * (abs(kkk[2]) < kmax_dealias),
        dtype=bool,
    )

    def rhs_fn(u, u_hat, rk):
        if rk > 0:
            u = jnp.fft.irfftn(u_hat, axes=fft_axes)

        curl = jnp.fft.irfftn(1j * jnp.cross(kkk, u_hat, axis=0), axes=fft_axes)
        du = jnp.fft.rfftn(jnp.cross(u.squeeze(), curl, axis=0), axes=fft_axes)
        du *= dealias
        p_hat = jnp.sum(du * kkk_over_kkk2, axis=0)
        du -= p_hat * kkk
        du -= nu * kkk2 * u_hat
        return du

    return rhs_fn


def rk4_wrapper(dt, rhs_fn, forcing_mask, e_kin_init, forcing_type, fft_axes=(1, 2, 3)):
    """Runge-Kutta 4th order integrator.

    Based on: https://github.com/spectralDNS/spectralDNS
    """
    a_rk4 = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]
    b_rk4 = [0.5, 0.5, 1.0]

    def step_fn(u, u_hat):
        u_temp1, u_temp2 = u_hat.copy(), u_hat.copy()
        for rk in range(4):  # RK4 integrator
            du = rhs_fn(u, u_hat, rk)
            if rk < 3:
                u_hat = u_temp1 + b_rk4[rk] * dt * du
            u_temp2 += a_rk4[rk] * dt * du
        u_hat = u_temp2

        ### Forcing for HIT case ###

        # Method taken from:
        # A. G. Lamorgese and D. A. Caughey and S. B. Pope, "Direct numerical
        # simulation of homogeneous turbulence with hyperviscosity", Physics of
        # Fluids, 17, 1, 015106, 2005, (https://doi.org/10.1063/1.1833415)

        # Implemented as spectralDNS does in their isotropic.py file

        e_inj = 0.0
        if forcing_mask is not None:
            if forcing_type == "ekin_tot":
                # current values
                u_new = jnp.fft.irfftn(u_hat, axes=fft_axes)
                e_kin_new = 0.5 * jnp.mean(jnp.sum(u_new * u_new, axis=0))

                # low-pass filter to keep only large scales (up to kf)
                u_lower = jnp.fft.irfftn(u_hat * forcing_mask, axes=fft_axes)
                e_kin_lower = 0.5 * jnp.mean(jnp.sum(u_lower * u_lower, axis=0))

                # high wavenumber energy to be removed
                e_kin_high = e_kin_new - e_kin_lower

                # scaling factor to add energy back to large scales
                alpha = jnp.sqrt(
                    jnp.maximum(0, (e_kin_init - e_kin_high) / (e_kin_lower + EPS))
                )

                # injection rate (for logging)
                e_inj = jnp.maximum(0, (e_kin_init - e_kin_new) / dt)

            elif forcing_type == "ekin_low":
                # low-pass filter to keep only large scales (up to kf)
                u_lower = jnp.fft.irfftn(u_hat * forcing_mask, axes=fft_axes)
                e_kin_lower = 0.5 * jnp.mean(jnp.sum(u_lower * u_lower, axis=0))

                # scaling factor to add energy back to large scales
                alpha = jnp.sqrt(jnp.maximum(0, e_kin_init / (e_kin_lower + EPS)))

                # injection rate (for logging)
                e_inj = jnp.maximum(0, (e_kin_init - e_kin_lower) / dt)

            # rescale large scales and transform back to spectral space
            u_hat *= alpha * forcing_mask + (1 - forcing_mask)

        u = jnp.fft.irfftn(u_hat, axes=fft_axes)
        if len(fft_axes) == 2:
            u = jnp.expand_dims(u, -1)

        return u, u_hat, e_inj

    return step_fn


def comp_dt(u, dx, nu, cfl=1.0):
    umax = jnp.max(jnp.abs(u), axis=(1, 2, 3))  # (3,)

    # Eq. (61) from ALDM paper, Hickel et al. (2006)
    dt = cfl / (umax / dx + nu / dx**2).min()
    return dt


def custom_jit_integrate(path_pref, rejit, integrate_fn, u, u_hat):
    if rejit or not os.path.exists(f"{path_pref}.bin"):  # 100s at 256^3
        u_shape = jax.ShapeDtypeStruct(u.shape, u.dtype)
        u_hat_shape = jax.ShapeDtypeStruct(u_hat.shape, u_hat.dtype)
        lowered = jax.jit(integrate_fn).lower(u_shape, u_hat_shape)
        integrate_fn = lowered.compile()
        serialized_bin, in_tree, out_tree = serialize_executable.serialize(integrate_fn)
        parent_dir = os.path.dirname(path_pref)
        os.makedirs(parent_dir, exist_ok=True)
        with open(f"{path_pref}.bin", "wb") as f:
            f.write(serialized_bin)
        with open(f"{path_pref}_tree.pkl", "wb") as f:
            pickle.dump((in_tree, out_tree), f)
        print("### Rejitted.")
    else:  # Load from file; 4s at 256^3
        with open(f"{path_pref}.bin", "rb") as f:
            serialized_bin = f.read()
        with open(f"{path_pref}_tree.pkl", "rb") as f:
            in_tree, out_tree = pickle.load(f)
        integrate_fn = serialize_executable.deserialize_and_load(
            serialized_bin, in_tree, out_tree
        )
        print("### Loaded.")
    return integrate_fn


def set_up_solver(
    N,
    nu,
    dim,
    case,
    dt,
    seed,
    ckp_N,
    kf,
    e_kin_target=1.0,
    forcing_type="ekin_tot",
    rejit=True,
):
    L = 2 * jnp.pi
    dx = L / N
    len_z = N if dim == 3 else 1
    len_z_vis = ckp_N if dim == 3 else 1
    fft_axes = (1, 2, 3) if dim == 3 else (1, 2)

    # a = jnp.mgrid[:N, :N, :len_z].astype(float) * L / N  # (3,N,N,N)
    xyz = jnp.meshgrid(jnp.arange(N), jnp.arange(N), jnp.arange(len_z), indexing="ij")
    xyz = jnp.array(xyz) * L / N
    x = jnp.arange(ckp_N)
    xyz_vis = jnp.meshgrid(x, x, jnp.arange(len_z_vis), indexing="ij")
    xyz_vis = jnp.array(xyz_vis) * L / ckp_N
    xyz_vis = (xyz_vis.T + jnp.array([0, 0, L - dx])).T if dim == 2 else xyz_vis
    # print(jnp.isclose(xyz,a).all(), a[:,1,0,0], xyz[:,1,0,0])

    # initialize forcing mask
    if case == "HIT" and forcing_type != "none":
        forcing_mask = init_forcing_mask_hit(N, kf)
    else:
        forcing_mask = None

    # initialize velocity field and its Fourier transform
    if case == "TGV":
        u = init_u_tgv(xyz[0], xyz[1], xyz[2])  # (3,N,N,N)
    elif case == "HIT":
        u = init_u_hit(N, seed)  # 7s at 256^3
    elif case == "Kolm":
        u = init_u_kolm(N, seed=seed, target_dim=3)
    elif case == "TGV2D":
        u = init_u_tgv2d(N, target_dim=3, rescale=L)

    # rescale initial velocity to match target kinetic energy
    e_kin_init = 0.5 * jnp.mean(jnp.sum(u * u, axis=0))
    u *= jnp.sqrt(e_kin_target / (e_kin_init + EPS))

    # transform to spectral space
    u_hat = jnp.fft.rfftn(u, axes=fft_axes).squeeze()  # (3,N,N,N//2+1)

    # compute initial kinetic energy as target for rescaling
    if forcing_type == "ekin_low":
        u_lower = jnp.fft.irfftn(u_hat * forcing_mask, axes=fft_axes)
        e_kin_init = 0.5 * jnp.mean(jnp.sum(u_lower * u_lower, axis=0))
    else:
        e_kin_init = 0.5 * jnp.mean(jnp.sum(u * u, axis=0))

    ######################################################################

    # import numpy as np
    # import matplotlib.pyplot as plt
    # from l3es.utils import spectral_filtering

    # u_ = np.asarray(u[:2, :, :, 0])
    # u_filtered = np.asarray(spectral_filtering(u_, ckp_N))
    # # apply a simple convolution averaging filter
    # # from scipy.signal import convolve2d
    # # u_filtered = []
    # # for i in range(2):
    # #     u_filtered.append(convolve2d(u_[i], np.ones((2, 2)) / 4, mode="valid", ))
    # # u_filtered = np.asarray(u_filtered)[:, ::2, ::2]

    # # print(div_in_spectral_space(u_), div_in_spectral_space(u_filtered))

    # def comp_divergence(u, L=2 * jnp.pi, version=1):
    #     """Numerically evaluate the divergence of a vector field."""
    #     res = np.zeros_like(u[0])
    #     N = u.shape[1]
    #     dx = dy = L / N
    #     print("#1", u.shape)
    #     if version == 1:
    #         res[1:-1,1:-1] = (
    #             u[0, 2:, 1:-1] - u[0, :-2, 1:-1]
    #             + u[1, 1:-1, 2:] - u[1, 1:-1, :-2]
    #         ) / (2 * dx)
    #     elif version == 2:
    #         temp = np.ufunc.reduce(np.add,
    #             [np.gradient(u[i], dx, axis=i) for i in range(len(u))])
    #         res[1:-1, 1:-1] = temp[1:-1, 1:-1]  # remove boundaries
    #     elif version == 3:  # set boundaries to zero
    #         # Extract x and y components of velocity
    #         u, v = u
    #         # Compute partial derivatives using central difference
    #         dudx = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dx)  # du/dx
    #         dvdy = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dy)  # dv/dy
    #         # Compute divergence in the central region of the grid
    #         res[1:-1, 1:-1] = dudx + dvdy
    #     elif version == 4:  # does not respect periodic boundary conditions
    #         dudx = np.gradient(u[0], dx, axis=0)
    #         dvdy = np.gradient(u[1], dx, axis=1)
    #         res[1:-1, 1:-1] = (dudx + dvdy)[1:-1, 1:-1]  # remove boundaries

    #     print("#2", res.shape)
    #     return res

    # fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    # plt.suptitle(f"Left ({N}x{N}) vs. Right ({ckp_N}x{ckp_N})")
    # for i, title in enumerate(["u_x", "u_y"]):
    #     axs[i, 0].imshow(u_[i], cmap="turbo")
    #     axs[i, 0].set_title(f"{title} (min={u_[i].min():.2f}, max={u_[i].max():.2f})")
    #     axs[i, 1].imshow(u_filtered[i], cmap="turbo")
    #     axs[i, 1].set_title(
    #         f"{title} (min={u_filtered[i].min():.2f}, max={u_filtered[i].max():.2f})")

    # version = 4
    # div_ = comp_divergence(u_, version=version)
    # axs[2, 0].imshow(div_, cmap="turbo")
    # axs[2, 0].set_title(f"Div (min={div_.min():.3f}, max={div_.max():.3f})")
    # div_filtered = comp_divergence(u_filtered, version=version)
    # axs[2, 1].imshow(div_filtered, cmap="turbo")
    # axs[2, 1].set_title(
    #     f"Div (min={div_filtered.min():.3f}, max={div_filtered.max():.3f})")

    # # after spectral filtering, field is not divergence-free
    # plt.savefig("divergence.png")
    # plt.close()
    ######################################################################

    t0 = time()
    integrate_fn = rk4_wrapper(
        dt,
        rhs_wrapper(N, nu, fft_axes),
        forcing_mask,
        e_kin_init,
        forcing_type,
        fft_axes,
    )
    device_type = "gpu" if jax.lib.xla_bridge.get_backend().platform == "gpu" else "cpu"
    if forcing_type != "none":
        path_pref = f".cache/spectral_{device_type}_{N}_{dim}_{kf}_{forcing_type}"
    else:
        path_pref = f".cache/spectral_{device_type}_{N}_{dim}"
    integrate_fn = custom_jit_integrate(path_pref, rejit, integrate_fn, u, u_hat)
    u, u_hat, e_inj = integrate_fn(u, u_hat)

    # import matplotlib.pyplot as plt
    # _, axs = plt.subplots(1, 2, figsize=(10, 5))
    # my_imshow(axs[0], u[0, :, :, 0], -u_ref, u_ref)
    # axs[1].scatter(xyz[0].reshape(-1), xyz[1].reshape(-1), c=u[0].reshape(-1),
    #     cmap="turbo", vmin=-u_ref, vmax=u_ref)
    # plt.savefig("orientation_check.png")

    u.block_until_ready()
    print(f"Compilation time {time() - t0:.3f}")

    return u, u_hat, xyz_vis, dx, integrate_fn, L, fft_axes, e_kin_init, e_inj


def simulate(
    case="TGV",
    N=64,
    dim=3,
    nu=0.000625,
    t_final=0.1,
    burnin=0,
    dt=0.01,
    u_ref=1.0,
    seed=42,
    log_freq=20,
    vis_freq=10**8,
    ckp_freq=1,
    ckp_N=32,
    dst_path=None,
    kf=3,
    forcing_type=None,
    e_kin_target=1.0,
    rejit=True,
):
    """Simulator wrapper.

    Args:
        case (str): Simulation case. One of ["TGV", "HIT", "Kolm", ...].
        N (int): Grid size. Nx=Ny=Nz=N. Simulation time:
            N=2**6=64  t_sim(steps=200)=1.46s, N=2**7=128 t_sim=3.55s, N=192 t_sim=22.2.
        dim (int): Dimension.
        nu (float): Viscosity = 1/Re. nu=0.000625 for Re=1600.
        t_final (float): Final time.
        burnin (int): Number of initial steps before full run starts.
        dt (float): Integration time step. N=64, Re=1600: dt=0.031 last stable; computed
            dt=0.027 (CFL=1.15); we use dt=0.01 (CFL=0.37).  N=192, Re=1600: computed
            dt=0.0087
        u_ref (float): Reference velocity. Used for plotting and CFL computation.
        e_kin_target (float): Target kinetic energy for forced HIT case.
        seed (int): Random seed for initialization.
        log_freq (int): How often to log simulation progress.
        vis_freq (int): How often to generate visualizations.
        ckp_freq (int): How often to save the flow field.
        ckp_N (int): How many spatial modes to keep (after spectral filtering).
        kf (int): Forcing up to wavenumber for HIT case.
        forcing_type (str): Forcing type for HIT case. One of:
            ["ekin_tot", "ekin_low", "none"].
            ekin_tot: rescale to keep total kinetic energy constant
            ekin_low: rescale only to keep low wavenumber kinetic energy constant.
            none: no forcing.
        rejit (bool): Whether to recompile the solver.
        dst_path (str): Where to write results. (Destination path)
    """

    u, u_hat, xyz_vis, dx, integrate_fn, L, _, e_kin_init, e_inj = set_up_solver(
        N, nu, dim, case, dt, seed, ckp_N, kf, e_kin_target, forcing_type, rejit
    )

    dst_vis = os.path.join(dst_path, "ckp_vis")
    dst_ckp = os.path.join(dst_path, "ckp")
    diagnostics_path = os.path.join(dst_path, "diagnostics.csv")

    with open(diagnostics_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "time", "umax", "ekin", "dt_est", "e_inj"])

    if vis_freq < 10**6:
        plot_e_k(u, 0, save_path=dst_vis, dim=dim, ylims=(1e-8, 1e2))
        if dim == 2:
            u_ckp = spectral_filtering(u[:2].squeeze(), ckp_N)[..., None]
        else:
            u_ckp = spectral_filtering(u, ckp_N)
        plot_views(xyz_vis, u_ckp, L / ckp_N, 0, save_path=dst_vis, u_ref=u_ref)
    if ckp_freq < 10**6:
        write_u(u, 0, dst_ckp, ckp_N)

    e_inj_acc = 0.0
    # e_diss_rate = 0.0

    print(
        f"{'#' * 79}\nSimulation with N={N}, nu={nu}, t_final={t_final}, dt={dt}, ",
        f"E_kin_init={e_kin_init:.3f}\n{'#' * 79}",
    )
    t_sim, t0 = 0, time()
    tstep_max = round(t_final / dt) + 1

    for i in range(tstep_max):
        if i == burnin:
            write_u(u, i, dst_path, N, suffix="_burnin")
        if i == burnin and case == "HIT":  # switch to a compiled solver without forcing
            _, _, _, _, integrate_fn, _, _, _, _ = set_up_solver(
                N, nu, dim, case, dt, seed, ckp_N, kf, e_kin_target, "none", rejit
            )
        t_temp = time()
        u, u_hat, e_inj = integrate_fn(u, u_hat)
        u.block_until_ready()
        t_sim += time() - t_temp

        # sum up injection rate to get eddy turnover time for forced HIT case
        e_inj_acc += e_inj

        # compute dissipation rate for logging
        # e_diss_rate += comp_dissipation_rate(u_hat, nu)

        if i % log_freq == 0:
            e_kin = 0.5 * float(jnp.mean(jnp.sum(u * u, axis=0)))
            umax = float(jnp.abs(u).max())
            dt_est = float(comp_dt(u, dx, nu))
            sim_time = float(i * dt)
            with open(diagnostics_path, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([i, sim_time, umax, e_kin, dt_est, float(e_inj)])
            print(
                f"step {i}, u_max = {umax:.3f}, E_kin = {e_kin:.3f}, "
                f"dt_est = {dt_est:.5f}, E_inj = {e_inj:.3f}"
            )

        if i % vis_freq == 0:
            if dim == 2:
                u_ckp = spectral_filtering(u[:2].squeeze(), ckp_N)[..., None]
            else:
                u_ckp = spectral_filtering(u, ckp_N)
            plot_views(xyz_vis, u_ckp, L / ckp_N, i, save_path=dst_vis, u_ref=u_ref)
            plot_e_k(u, i, save_path=dst_vis, dim=dim, ylims=(1e-8, 1e2))
        if i % ckp_freq == 0:
            write_u(u, i, dst_ckp, ckp_N)

    if case == "HIT" and forcing_type != "none":
        hit_eddy_turnover_time = e_inj_acc / i
        hit_eddy_turnover_time *= kf**2
        hit_eddy_turnover_time = 1.0 / hit_eddy_turnover_time ** (1 / 3)
        print(f"Eddy turnover time =            {hit_eddy_turnover_time:.3f}")
        print(f"Number of eddy turnover times = {t_final / hit_eddy_turnover_time:.3f}")
        # print(f"Average energy dissipated =     {e_diss_rate / tstep:.3f}")
        print(f"Average energy injected =       {e_inj_acc / i:.3f}")

        kmax = jnp.sqrt(2) * N / 3
        kolm_scale = (nu**3 / (e_inj_acc / i)) ** (0.25)
        print(f"eta=                            {kolm_scale:.3f}")
        print(f"kmax*eta =                      {kmax*kolm_scale:.3f}")

        Re_lambda = jnp.sqrt(20 * e_kin**2 / (3 * nu * e_inj_acc / i))
        print(f"Re_lambda =                     {Re_lambda:.3f}")

    t_tot = time() - t0
    print(f"t_tot = {t_tot:.3f}, t_sim = {t_sim:.3f}")

    # Validation case with reference kinetic energy value.
    if t_final == 0.1 and case == "TGV":
        e_kin = 0.5 * jnp.mean(jnp.sum(u * u, axis=0))
        assert round(e_kin - 0.124953117517, 2) == 0, f"e_kin = {e_kin}"
        print("Assert passed!")
