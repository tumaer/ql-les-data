import jax.numpy as jnp
from jax import config

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
