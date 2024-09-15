import jax.numpy as jnp
import jax_cfd.base as cfd  # version 0.2.1
import numpy as np
from jax import random
from jax_cfd.collocated import initial_conditions

EPS = jnp.finfo(float).eps


def init_u_tgv(x, y, z):
    u = jnp.array(
        [np.sin(x) * np.cos(y) * np.cos(z), -np.cos(x) * np.sin(y) * np.cos(z), x * 0]
    )
    return u


def init_u_tgv2d(Nx, target_dim=2, rescale=1.0):
    # rescale is used for the velocity and the domain size
    L = rescale
    dx = L / Nx
    x_axis = np.linspace(dx / 2, L - dx / 2, Nx)
    x, y = np.meshgrid(x_axis, x_axis)
    u = np.array([-np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)])

    if target_dim == 3:
        u = u[..., None]
        u = jnp.concatenate([u, jnp.zeros((1, Nx, Nx, 1))], axis=0)
    return u * rescale


def init_u_hit(N, seed=42):
    np.random.seed(seed)

    def apply_conjugate_symmetry_2D(A):
        N = A.shape[0]
        Nf_ = N // 2
        A[-Nf_ + 1 :, -Nf_ + 1 :] = np.flip(np.conj(A[1:Nf_, 1:Nf_]), axis=(0, 1))
        A[1:Nf_, -Nf_ + 1 :] = np.flip(np.conj(A[-Nf_ + 1 :, 1:Nf_]), axis=(0, 1))
        A[0, -Nf_ + 1 :] = np.flip(np.conj(A[0, 1:Nf_]))
        A[-Nf_ + 1 :, 0] = np.flip(np.conj(A[1:Nf_, 0]))
        return A

    u = np.zeros((N // 2 + 1, N, N, 3), dtype=np.complex128)
    u[:, :, :, :] = 0.0
    nmax = round(np.sqrt((N**2 + N**2 + N**2)) / 2)

    # initialize E(k)
    E_k = np.zeros(nmax + 1) + EPS

    for k in range(1, nmax + 1):
        E_k[k] = k ** (-5 / 3) / (4 * np.pi * k**2)

    for k1 in range(N // 2 + 1):  # 0-16
        i = k1
        for k2 in range(-N // 2, N // 2):
            j = (N + k2) % N
            for k3 in range(-N // 2, N // 2):
                k = (N + k3) % N
                wk = np.sqrt(k1**2 + k2**2 + k3**2)
                wk12 = np.sqrt(k1**2 + k2**2)
                nk = round(wk)
                amp = np.sqrt(E_k[nk])
                # 0.948142290  # np.random.random()  # uniform in [0, 2pi)
                phi1, phi2, phi3 = 2 * np.pi * np.random.random(size=3)
                ai = amp * (np.cos(phi1) + np.sin(phi1) * 1j) * np.cos(phi3)
                bi = amp * (np.cos(phi2) + np.sin(phi2) * 1j) * np.sin(phi3)
                if wk != 0:
                    if wk12 != 0.0:
                        u[i, j, k, 0] = (ai * wk * k2 + bi * k1 * k3) / (wk * wk12)
                        u[i, j, k, 1] = (bi * k2 * k3 - ai * wk * k1) / (wk * wk12)
                    else:
                        u[i, j, k, 0] = ai
                        u[i, j, k, 1] = bi
                    u[i, j, k, 2] = -bi * wk12 / wk
                else:
                    u[i, j, k, 0] = 0.0
                    u[i, j, k, 1] = 0.0
                    u[i, j, k, 2] = 0.0

    # enforce symmetry of matrix
    for ii in range(3):
        u[0, :, :, ii] = apply_conjugate_symmetry_2D(u[0, :, :, ii])

    # transform to real space
    u = jnp.fft.irfftn(u.transpose(3, 0, 1, 2), axes=(-1, -2, -3), norm="forward")
    return u


def init_u_kolm(N, L=2 * np.pi, max_velocity=7, seed=42, target_dim=3, iter=4):
    """Kolgomorov 2D field initialization. Based on
    https://github.com/google/jax-cfd/blob/0c17e3855702f884265b97bd6ff0793c34f3155e/jax_cfd/collocated/initial_conditions_test.py

    The `iter` that work well in 2D are [4, 9, 12, 13, 18, ...]. Not sure why
    the default is 3. Some intermediate values of `iter` blow up completely!?
    """
    grid = cfd.grids.Grid((N, N), domain=((0, L), (0, L)))

    # Fails: from jax_cfd.base import initial_conditions
    # Works: from jax_cfd.collocated import initial_conditions
    v0 = initial_conditions.filtered_velocity_field(
        random.PRNGKey(seed), grid, max_velocity, 4, iterations=iter
    )
    v0 = jnp.array([v.data for v in v0])[..., None]  # (2, N, N, 1)

    if target_dim == 3:
        v0 = jnp.concatenate([v0, jnp.zeros((1, N, N, 1))], axis=0)
    return v0
