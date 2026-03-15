import jax.numpy as jnp
import jax_cfd.base as cfd  # version 0.2.1
import numpy as np
from jax import random
from jax_cfd.collocated import initial_conditions

from l3es.utils import spectral_filtering

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
    """Vectorized version of `init_u_hit_slow`."""
    np.random.seed(seed)

    # 1. Precompute Spectrum (E_k)
    nmax = round(np.sqrt(3 * N**2) / 2)
    E_k = np.zeros(nmax + 1) + EPS
    k_arr = np.arange(1, nmax + 1)
    E_k[1:] = k_arr ** (-5 / 3) / (4 * np.pi * k_arr**2)

    # 2. Setup grid matching the original loop execution order
    k1_loop = np.arange(N // 2 + 1)
    k2_loop = np.arange(-N // 2, N // 2)
    k3_loop = np.arange(-N // 2, N // 2)

    K1, K2, K3 = np.meshgrid(k1_loop, k2_loop, k3_loop, indexing="ij")

    WK = np.sqrt(K1**2 + K2**2 + K3**2)
    WK12 = np.sqrt(K1**2 + K2**2)
    NK = np.clip(np.round(WK).astype(int), 0, nmax)
    AMP = np.sqrt(E_k[NK])

    # 3. Vectorized RNG matching exact original sequence
    phis = 2 * np.pi * np.random.random((N // 2 + 1, N, N, 3))
    phi1, phi2, phi3 = phis[..., 0], phis[..., 1], phis[..., 2]

    # Complex amplitudes
    ai = AMP * np.exp(1j * phi1) * np.cos(phi3)
    bi = AMP * np.exp(1j * phi2) * np.sin(phi3)

    # 4. Compute components safely (avoiding zero division warnings)
    WK_safe = np.where(WK == 0, 1.0, WK)
    WK12_safe = np.where(WK12 == 0, 1.0, WK12)

    u_loop = np.zeros((N // 2 + 1, N, N, 3), dtype=np.complex128)

    # Base velocity field calculation
    u_loop[..., 0] = (ai * WK * K2 + bi * K1 * K3) / (WK_safe * WK12_safe)
    u_loop[..., 1] = (bi * K2 * K3 - ai * WK * K1) / (WK_safe * WK12_safe)
    u_loop[..., 2] = -bi * WK12 / WK_safe

    # Apply conditions: wk12 == 0
    mask_wk12_zero = WK12 == 0
    u_loop[..., 0] = np.where(mask_wk12_zero, ai, u_loop[..., 0])
    u_loop[..., 1] = np.where(mask_wk12_zero, bi, u_loop[..., 1])

    # Apply conditions: wk == 0
    mask_wk_zero = WK == 0
    u_loop[..., 0] = np.where(mask_wk_zero, 0.0, u_loop[..., 0])
    u_loop[..., 1] = np.where(mask_wk_zero, 0.0, u_loop[..., 1])
    u_loop[..., 2] = np.where(mask_wk_zero, 0.0, u_loop[..., 2])

    # 5. Remap from loop-order to standard FFT spatial mapping (j, k indices)
    j_idx = (N + k2_loop) % N
    k_idx = (N + k3_loop) % N
    j_inv = np.argsort(j_idx)
    k_inv = np.argsort(k_idx)

    u = u_loop[:, j_inv, :][:, :, k_inv]

    # 6. Enforce conjugate symmetry in a vectorized manner
    Nf_ = N // 2
    u[0, -Nf_ + 1 :, -Nf_ + 1 :, :] = np.flip(
        np.conj(u[0, 1:Nf_, 1:Nf_, :]), axis=(0, 1)
    )
    u[0, 1:Nf_, -Nf_ + 1 :, :] = np.flip(
        np.conj(u[0, -Nf_ + 1 :, 1:Nf_, :]), axis=(0, 1)
    )
    u[0, 0, -Nf_ + 1 :, :] = np.flip(np.conj(u[0, 0, 1:Nf_, :]), axis=0)
    u[0, -Nf_ + 1 :, 0, :] = np.flip(np.conj(u[0, 1:Nf_, 0, :]), axis=0)

    # 7. Transform to real space
    u = jnp.fft.irfftn(u.transpose(3, 0, 1, 2), axes=(-1, -2, -3), norm="forward")
    return u


def init_u_hit_slow(N, seed=42):
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


def init_forcing_mask_hit(N, kf=3):
    """Initialize a wavenumber mask for constant energy forcing in HIT."""

    # 1D wavenumber grid
    k1 = jnp.fft.fftfreq(N, 1.0 / N)

    # since we use rfftn, the last axis is truncated to N//2+1
    k3 = k1[: N // 2 + 1].copy()
    k3 = k3.at[-1].set(
        -1 * k3[-1]
    )  # set the last element to negative to get correct wavenumber

    # 3D wavenumber grid
    k1k2k3 = jnp.asarray(jnp.meshgrid(k1, k1, k3, indexing="ij"))

    # compute squared wavenumber magnitude
    k_mag = jnp.sum(k1k2k3**2, axis=0)

    # create mask for wavenumbers in the forcing range
    mask = (k_mag <= kf**2) & (k_mag > 0)

    return mask


def init_u_kolm(N, L=2 * np.pi, max_velocity=4.2, seed=42, target_dim=3, iter=4):
    """Kolgomorov 2D field initialization. Based on
    https://github.com/google/jax-cfd/blob/0c17e3855702f884265b97bd6ff0793c34f3155e/jax_cfd/collocated/initial_conditions_test.py

    The `iter` that work well in 2D are [4, 9, 12, 13, 18, ...]. Not sure why
    the default is 3. Some intermediate values of `iter` blow up completely!?
    """
    grid = cfd.grids.Grid((2048, 2048), domain=((0, L), (0, L)))

    # Fails: from jax_cfd.base import initial_conditions
    # Works: from jax_cfd.collocated import initial_conditions
    v0 = initial_conditions.filtered_velocity_field(
        random.PRNGKey(seed), grid, max_velocity, 4, iterations=iter
    )
    v0 = jnp.array([v.data for v in v0])[..., None]  # (2, N, N, 1)

    # spectrally filter down to N x N
    v0 = spectral_filtering(v0.squeeze(), N)[..., None]  # (2, N, N, 1)

    if target_dim == 3:
        v0 = jnp.concatenate([v0, jnp.zeros((1, N, N, 1))], axis=0)  # (3, N, N, 1)
    return v0
