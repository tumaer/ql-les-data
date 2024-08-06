from time import time

import jax.numpy as jnp
import numpy as np
from jax import config, jit

from l3es.init_fields import init_u_hit, init_u_tgv
from l3es.spectral_solver import comp_dt, rhs_wrapper, rk4_wrapper
from l3es.visualize import plot_e_k, plot_views

EPS = jnp.finfo(float).eps
config.update("jax_enable_x64", True)


if __name__ == "__main__":
    nu = 0.000625  # Re=1600
    t_final = 2.0  # 0.1
    # N=64, Re=1600: dt=0.031 last stable. Computed dt=0.027 (CFL=1.15).
    # N=64, Re=1600: We use dt=0.01 (CFL=0.37)
    # N=196, Re=1600: Computed dt=0.0087
    dt = 0.01
    N = 2**7  # 2**6=64  t_sim(steps=200)=1.46s, 2**7=128 t_sim=3.55s, 196 t_sim=22.2s
    dx = 2 * np.pi / N
    case = "HIT"
    vis = False

    xyz = jnp.mgrid[:N, :N, :N].astype(float) * 2 * jnp.pi / N  # (3,N,N,N)
    if case == "TGV":
        u = init_u_tgv(xyz[0], xyz[1], xyz[2])  # (3,N,N,N)
    elif case == "HIT":
        u = init_u_hit(N)
    u_hat = jnp.fft.rfftn(u, axes=(1, 2, 3))  # (3,N,N,N//2+1)

    integrate_fn = rk4_wrapper(dt, rhs_wrapper(N, nu))
    integrate_fn = jit(integrate_fn)
    u, u_hat = integrate_fn(u, u_hat)
    u.block_until_ready()

    t = 0.0
    tstep = 0
    if vis:
        plot_e_k(u, tstep, save_path="results")
        plot_views(xyz, u, dx, tstep, save_path="results")
    t0 = time()
    t_vis = 0.0
    while t < t_final - 1e-8:
        t += dt
        tstep += 1
        u, u_hat = integrate_fn(u, u_hat)
        if tstep % 20 == 0:
            t_temp = time()
            if vis:
                plot_views(xyz, u, dx, tstep, save_path="results")
                plot_e_k(u, tstep, save_path="results")
            print(
                f"step {tstep}, u_max = {abs(u).max()}, dt_est = {comp_dt(u, dx, nu)}"
            )
            t_vis += time() - t_temp

    t_tot = time() - t0
    print(f"t_tot = {t_tot:.2f}, t_sim = {t_tot-t_vis:.2f}")

    e_kin = 0.5 * jnp.mean(jnp.sum(u * u, axis=0))
    if t_final == 0.1 and case == "TGV":
        assert round(e_kin - 0.124953117517, 2) == 0, f"e_kin = {e_kin}"
        print("Assert passed")
