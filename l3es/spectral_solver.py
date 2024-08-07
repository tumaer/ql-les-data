"""Core spectral DNS solver."""

import os
from time import time

import jax.numpy as jnp
import numpy as np
from jax import config, jit

from l3es.init_fields import init_u_hit, init_u_tgv
from l3es.utils import write_u
from l3es.visualize import plot_e_k, plot_views

EPS = jnp.finfo(float).eps
config.update("jax_enable_x64", True)


def rhs_wrapper(N, nu):
    """Spectral Navier-Stokes solver right hand side.

    Based on: https://github.com/spectralDNS/spectralDNS
    """
    kx = jnp.fft.fftfreq(N, 1.0 / N)
    kz = kx[: (N // 2 + 1)].copy()
    # kz[-1] *= -1
    kz = kz.at[-1].set(-1 * kz[-1])
    kkk = jnp.array(jnp.meshgrid(kx, kx, kz, indexing="ij"), dtype=int)
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
            u = jnp.fft.irfftn(u_hat, axes=(1, 2, 3))

        curl = jnp.fft.irfftn(
            1j * jnp.cross(kkk, u_hat, axis=0), axes=(1, 2, 3)
        )  # Curl
        du = jnp.fft.rfftn(jnp.cross(u, curl, axis=0), axes=(1, 2, 3))  # Cross
        du *= dealias
        p_hat = jnp.sum(du * kkk_over_kkk2, axis=0)
        du -= p_hat * kkk
        du -= nu * kkk2 * u_hat
        return du

    return rhs_fn


def rk4_wrapper(dt, rhs_fn):
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
        u = jnp.fft.irfftn(u_hat, axes=(1, 2, 3))
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
    nu=0.000625,
    t_final=0.1,
    dt=0.01,
    seed=42,
    log_freq=20,
    vis_freq=10**8,
    ckp_freq=1,
    ckp_N=32,
    dst_path=None,
):
    """Simulator wrapper.

    Args:
        N (int): Grid size. Nx=Ny=Nz=N. Simulation time:
            N=2**6=64  t_sim(steps=200)=1.46s, N=2**7=128 t_sim=3.55s, N=196 t_sim=22.2.
        nu (float): Viscosity = 1/Re. nu=0.000625 for Re=1600.
        t_final (float): Final time.
        dt (float): Integration time step. N=64, Re=1600: dt=0.031 last stable; computed
            dt=0.027 (CFL=1.15); we use dt=0.01 (CFL=0.37).  N=196, Re=1600: computed
            dt=0.0087
        case (str): Simulation case. One of ["TGV", "HIT"].
        log_freq (int): How often to log simulation progress.
        vis_freq (int): How often to generate visualizations.
        ckp_freq (int): How often to save the flow field.
        ckp_N (int): How many spatial modes to keep (after spectral filtering).
        dst_path (str): Where to write results. (Destination path)
    """

    dx = 2 * np.pi / N

    xyz = jnp.mgrid[:N, :N, :N].astype(float) * 2 * jnp.pi / N  # (3,N,N,N)
    if case == "TGV":
        u = init_u_tgv(xyz[0], xyz[1], xyz[2])  # (3,N,N,N)
    elif case == "HIT":
        u = init_u_hit(N, seed)
    u_hat = jnp.fft.rfftn(u, axes=(1, 2, 3))  # (3,N,N,N//2+1)

    integrate_fn = rk4_wrapper(dt, rhs_wrapper(N, nu))
    integrate_fn = jit(integrate_fn)
    u, u_hat = integrate_fn(u, u_hat)
    u.block_until_ready()

    dst_vis = os.path.join(dst_path, "vis")
    dst_ckp = os.path.join(dst_path, "ckp")
    if vis_freq < 10**6:
        plot_e_k(u, 0, save_path=dst_vis)
        plot_views(xyz, u, dx, 0, save_path=dst_vis)
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
            plot_views(xyz, u, dx, tstep, save_path=dst_vis)
            plot_e_k(u, tstep, save_path=dst_vis)
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
