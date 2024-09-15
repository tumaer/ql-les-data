"""Core spectral DNS solver."""

import os
from time import time

import jax.numpy as jnp
from jax import config, jit

from l3es.init_fields import init_u_hit, init_u_kolm, init_u_tgv, init_u_tgv2d
from l3es.utils import write_u
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


def rk4_wrapper(dt, rhs_fn, fft_axes=(1, 2, 3)):
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
        u = jnp.fft.irfftn(u_hat, axes=fft_axes)
        if len(fft_axes) == 2:
            u = jnp.expand_dims(u, -1)
        return u, u_hat

    return step_fn


def comp_dt(u, dx, nu, cfl=1.0):
    umax = jnp.max(jnp.abs(u), axis=(1, 2, 3))  # (3,)

    # Eq. (61) from ALDM paper, Hickel et al. (2006)
    dt = cfl / (umax / dx + nu / dx**2).min()
    return dt


def simulate(
    case="TGV",
    N=64,
    dim=3,
    nu=0.000625,
    t_final=0.1,
    dt=0.01,
    u_ref=1.0,
    seed=42,
    log_freq=20,
    vis_freq=10**8,
    ckp_freq=1,
    ckp_N=32,
    dst_path=None,
):
    """Simulator wrapper.

    Args:
        case (str): Simulation case. One of ["TGV", "HIT", "Kolm", ...].
        N (int): Grid size. Nx=Ny=Nz=N. Simulation time:
            N=2**6=64  t_sim(steps=200)=1.46s, N=2**7=128 t_sim=3.55s, N=192 t_sim=22.2.
        dim (int): Dimension.
        nu (float): Viscosity = 1/Re. nu=0.000625 for Re=1600.
        t_final (float): Final time.
        dt (float): Integration time step. N=64, Re=1600: dt=0.031 last stable; computed
            dt=0.027 (CFL=1.15); we use dt=0.01 (CFL=0.37).  N=192, Re=1600: computed
            dt=0.0087
        u_ref (float): Reference velocity. Used for plotting and CFL computation.
        case (str): Simulation case. One of ["TGV", "HIT"].
        log_freq (int): How often to log simulation progress.
        vis_freq (int): How often to generate visualizations.
        ckp_freq (int): How often to save the flow field.
        ckp_N (int): How many spatial modes to keep (after spectral filtering).
        dst_path (str): Where to write results. (Destination path)
    """

    L = 2 * jnp.pi
    dx = L / N
    len_z = N if dim == 3 else 1
    fft_axes = (1, 2, 3) if dim == 3 else (1, 2)

    # a = jnp.mgrid[:N, :N, :len_z].astype(float) * L / N  # (3,N,N,N)
    xyz = jnp.meshgrid(jnp.arange(N), jnp.arange(N), jnp.arange(len_z), indexing="ij")
    xyz = jnp.array(xyz) * L / N
    xyz_vis = (xyz.T + jnp.array([0, 0, L - dx])).T if dim == 2 else xyz
    # print(jnp.isclose(xyz,a).all(), a[:,1,0,0], xyz[:,1,0,0])

    if case == "TGV":
        u = init_u_tgv(xyz[0], xyz[1], xyz[2])  # (3,N,N,N)
    elif case == "HIT":
        u = init_u_hit(N, seed)
    elif case == "Kolm":
        u = init_u_kolm(N, target_dim=3)
    elif case == "TGV2D":
        u = init_u_tgv2d(N, target_dim=3, rescale=L)
    u_hat = jnp.fft.rfftn(u, axes=fft_axes).squeeze()  # (3,N,N,N//2+1)

    ######################################################################

    # import numpy as np
    # import matplotlib.pyplot as plt
    # from l3es.utils import spectral_filtering

    # u_ = np.asarray(u[:2, :, :, 0])
    # u_filtered = np.asarray(spectral_filtering(u_, ckp_N))
    # print(div_in_spectral_space(u_), div_in_spectral_space(u_filtered))

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
    #         res = np.ufunc.reduce(np.add, [np.gradient(u[i], dx, axis=i) for i in range(len(u))])
    #     elif version == 3:
    #         # Extract x and y components of velocity
    #         u, v = u
    #         # Compute partial derivatives using central difference
    #         dudx = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dx)  # du/dx
    #         dvdy = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dy)  # dv/dy
    #         # Compute divergence in the central region of the grid
    #         res = np.zeros_like(u)
    #         res[1:-1, 1:-1] = dudx + dvdy
    #     elif version == 4:
    #         dudx = np.gradient(u[0], dx, axis=0)
    #         dvdy = np.gradient(u[1], dx, axis=1)
    #         res = dudx + dvdy

    #     print("#2", res.shape)
    #     return res

    # fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    # plt.suptitle(f"Left ({N}x{N}) vs. Right ({ckp_N}x{ckp_N})")
    # for i, title in enumerate(["u_x", "u_y"]):
    #     axs[i, 0].imshow(u_[i], cmap="turbo")
    #     axs[i, 0].set_title(f"{title} (min={u_[i].min():.2f}, max={u_[i].max():.2f})")
    #     axs[i, 1].imshow(u_filtered[i], cmap="turbo")
    #     axs[i, 1].set_title(f"{title} (min={u_filtered[i].min():.2f}, max={u_filtered[i].max():.2f})")

    # version = 4
    # div_ = comp_divergence(u_, version=version)
    # axs[2, 0].imshow(div_, cmap="turbo")
    # axs[2, 0].set_title(f"Div (min={div_.min():.2f}, max={div_.max():.2f})")
    # div_filtered = comp_divergence(u_filtered, version=version)
    # axs[2, 1].imshow(div_filtered, cmap="turbo")
    # axs[2, 1].set_title(f"Div (min={div_filtered.min():.2f}, max={div_filtered.max():.2f})")

    # plt.savefig("divergence.png")
    # plt.close()
    ######################################################################

    t0 = time()
    integrate_fn = rk4_wrapper(dt, rhs_wrapper(N, nu, fft_axes), fft_axes)
    integrate_fn = jit(integrate_fn)
    u, u_hat = integrate_fn(u, u_hat)

    # import matplotlib.pyplot as plt
    # _, axs = plt.subplots(1, 2, figsize=(10, 5))
    # my_imshow(axs[0], u[0, :, :, 0], -u_ref, u_ref)
    # axs[1].scatter(xyz[0].reshape(-1), xyz[1].reshape(-1), c=u[0].reshape(-1), cmap="turbo", vmin=-u_ref, vmax=u_ref)
    # plt.savefig("orientation_check.png")

    u.block_until_ready()
    print("Compilation time:", time() - t0)

    dst_vis = os.path.join(dst_path, "vis")
    dst_ckp = os.path.join(dst_path, "ckp")
    if vis_freq < 10**6:
        plot_e_k(u, 0, save_path=dst_vis, dim=dim)
        plot_views(xyz_vis, u, dx, 0, save_path=dst_vis, u_ref=u_ref)
    if ckp_freq < 10**6:
        write_u(u, 0, dst_ckp, ckp_N)

    print("#" * 79, f"\nSimulation with N={N}, nu={nu}, t_final={t_final}, dt={dt}")
    t = 0.0
    tstep = 0
    t0 = time()
    t_out = 0.0
    while t < t_final - 1e-8:
        t += dt
        tstep += 1
        u, u_hat = integrate_fn(u, u_hat)

        t_temp = time()
        if tstep % log_freq == 0:
            print(
                f"step {tstep}, u_max = {abs(u).max():.3f}, "
                f"dt_est = {comp_dt(u, dx, nu):.5f}"
            )
        if tstep % vis_freq == 0:
            plot_views(xyz_vis, u, dx, tstep, save_path=dst_vis, u_ref=u_ref)
            plot_e_k(u, tstep, save_path=dst_vis, dim=dim)
        if tstep % ckp_freq == 0:
            write_u(u, tstep, dst_ckp, ckp_N)
        t_out += time() - t_temp

    t_tot = time() - t0
    print(f"t_tot = {t_tot:.3f}, t_sim = {t_tot-t_out:.3f}")

    # Validation case with reference kinetic energy value.
    if t_final == 0.1 and case == "TGV":
        e_kin = 0.5 * jnp.mean(jnp.sum(u * u, axis=0))
        assert round(e_kin - 0.124953117517, 2) == 0, f"e_kin = {e_kin}"
        print("Assert passed!")
