from time import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftfreq, irfftn, rfftn


def plot_views(xyz, u, dx, step):
    x, y, z = xyz

    mask = (x > 2 * np.pi - 1.5 * dx) + (y < 0.5 * dx) + (z > 2 * np.pi - 1.5 * dx)

    def subplot_i(fig, ind, c, lbl):
        ax = fig.add_subplot(1, 4, ind, projection="3d")
        ax.view_init(elev=25.0, azim=-35, roll=0)
        ax.scatter(x[mask], y[mask], z[mask], c=c[mask], cmap="turbo")
        ax.set_aspect("equal", "box")
        ax.set_title(lbl)

    # plot results
    fig = plt.figure(figsize=(20, 5))
    fields = [u[0], u[1], u[2], np.linalg.norm(u, axis=0)]
    labels = ["ux", "uy", "uz", "|u|"]
    for i, (c, lbl) in enumerate(zip(fields, labels)):
        subplot_i(fig, i + 1, c, lbl)

    plt.savefig(f"results/step_{step}.png")


def init_u_tgv(x, y, z):
    u = np.array(
        [np.sin(x) * np.cos(y) * np.cos(z), -np.cos(x) * np.sin(y) * np.cos(z), x * 0]
    )
    return u


def rhs_wrapper(N):
    """Spectral Navier-Stokes solver right hand side."""
    kx = fftfreq(N, 1.0 / N)
    kz = kx[: (N // 2 + 1)].copy()
    kz[-1] *= -1
    kkk = np.array(np.meshgrid(kx, kx, kz, indexing="ij"), dtype=int)
    kkk2 = np.sum(kkk * kkk, 0, dtype=int)
    kkk_over_kkk2 = kkk.astype(float) / np.where(kkk2 == 0, 1, kkk2).astype(float)
    kmax_dealias = 2.0 / 3.0 * (N // 2 + 1)
    dealias = np.array(
        (abs(kkk[0]) < kmax_dealias)
        * (abs(kkk[1]) < kmax_dealias)
        * (abs(kkk[2]) < kmax_dealias),
        dtype=bool,
    )

    def cross_fn(a, b):
        return np.array(
            [
                rfftn(a[1] * b[2] - a[2] * b[1]),
                rfftn(a[2] * b[0] - a[0] * b[2]),
                rfftn(a[0] * b[1] - a[1] * b[0]),
            ]
        )

    def curl_fn(a):
        return np.array(
            [
                irfftn(1j * (kkk[1] * a[2] - kkk[2] * a[1])),
                irfftn(1j * (kkk[2] * a[0] - kkk[0] * a[2])),
                irfftn(1j * (kkk[0] * a[1] - kkk[1] * a[0])),
            ]
        )

    def rhs_fn(u, u_hat, rk):
        if rk > 0:
            u = irfftn(u_hat, axes=(1, 2, 3))

        # curl = irfftn(1j*np.cross(kkk, u_hat, axis=0), axes=(1,2,3))  # Curl
        # du = rfftn(np.cross(u, curl, axis=0), axes=(1,2,3))  # Cross
        curl = curl_fn(u_hat)
        du = cross_fn(u, curl)
        du *= dealias
        p_hat = np.sum(du * kkk_over_kkk2, axis=0)
        du -= p_hat * kkk
        du -= nu * kkk2 * u_hat
        return du

    return rhs_fn


def rk4_wrapper(dt, rhs_fn):
    """Runge-Kutta 4th order integrator."""
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
        u = irfftn(u_hat, axes=(1, 2, 3))
        return u, u_hat

    return step_fn


if __name__ == "__main__":
    nu = 0.000625
    t_final = 0.1  # 0.1
    dt = 0.01
    N = 2**6  # 64
    dx = 2 * np.pi / N

    xyz = np.mgrid[:N, :N, :N].astype(float) * 2 * np.pi / N  # (3,N,N,N)
    u = init_u_tgv(xyz[0], xyz[1], xyz[2])  # (3,N,N,N)
    u_hat = rfftn(u, axes=(1, 2, 3))  # (3,N,N,N//2+1)

    integrate_fn = rk4_wrapper(dt, rhs_wrapper(N))

    t = 0.0
    tstep = 0
    t0 = time()
    while t < t_final - 1e-8:
        t += dt
        tstep += 1
        u, u_hat = integrate_fn(u, u_hat)

    # plot_views(xyz, U, dx, tstep)
    print("Time = {}".format(time() - t0))

    e_kin = 0.5 * np.mean(np.sum(u * u, axis=0))
    if t_final == 0.1:
        assert round(e_kin - 0.124953117517, 7) == 0
        print("Assert passed")
