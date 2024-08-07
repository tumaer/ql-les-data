"""Utilities for L3ES."""

import os

import jax.numpy as jnp
import numpy as np

EPS = jnp.finfo(float).eps


def get_real_wavenumber_grid(n, dim):
    """Initializes wavenumber grid and wavenumber vector.

    Code based on JAX-FLUIDS implementation."""

    Nf = n // 2 + 1
    k = np.fft.fftfreq(n, 1.0 / n)  # for other dimensions
    kx = k[:Nf].copy()
    kx[-1] *= -1
    if dim == 2:
        k_field = jnp.array(np.meshgrid(kx, k, indexing="ij"), dtype=int)
    elif dim == 3:
        k_field = jnp.array(np.meshgrid(kx, k, k, indexing="ij"), dtype=int)
    k_vec = jnp.arange(n)
    return k_field, k_vec


def energy_spectrum(vel, mul_fac: float = 1.0, is_scalar_field: bool = False):
    """JAX implemented energy spectrum computation on a grid.

    Code based on JAX-FLUIDS 1.0 implementation."""

    dim = vel.shape[0]
    ns = vel.shape[1:]

    # check for square box with equal side length
    assert jnp.array_equal(ns, jnp.ones(dim) * ns[0])

    # common resolution
    n = ns[0]

    # Fourier transform
    if dim == 1:
        raise NotImplementedError("1D not implemented")
    elif dim == 2:
        vel_hat = jnp.fft.rfftn(vel, axes=(2, 1))
    elif dim == 3:
        vel_hat = jnp.fft.rfftn(vel, axes=(3, 2, 1))

    # initialize wavenumber grid
    k_field, k = get_real_wavenumber_grid(n, dim)  # (3,17,32,32), (0,....31)

    # compute prefactor
    fact = (
        2 * (k_field[0] > 0) * (k_field[0] < n // 2)
        + 1 * (k_field[0] == 0)
        + 1 * (k_field[0] == n // 2)
    )  # (17,32,32) bw 1-2

    # calculate wavenumber vector norms
    k_field_norm = jnp.linalg.norm(k_field, axis=0, ord=2)  # (17,32,32)

    # calculate integration shell
    shell = (k_field_norm + 0.5).astype(int).flatten()

    # fourier transform prefactor
    vel_hat /= n**3

    # calculate energy
    abs_energy = jnp.sum(jnp.abs(vel_hat**2), axis=0)
    abs_energy *= fact * mul_fac

    # number of samples
    n_samples = jnp.zeros(n)
    n_samples = n_samples.at[shell].add(fact.flatten())

    # compute energy spectrum
    ek = jnp.zeros(n)
    ek = ek.at[shell].add(abs_energy.flatten())
    ek *= 4 * jnp.pi * k**2 / (n_samples + EPS)

    return ek


def write_u(u, tstep, dst_path, ckp_N):
    """Write flow field to disk.

    Args:
        u (jnp.ndarray): Flow field with shape (3, N, N, N).
        tstep (int): Current time step.
        dst_path (str): Where to write results.
        ckp_N (int): How many spatial modes to keep (after spectral filtering).
    """

    N_u = u.shape[1]
    assert ckp_N <= N_u, "ckp_N must be less than or equal to N."
    assert ckp_N % 2 == 0 and N_u % 2 == 0, "Only tested for even N and ckp_N."

    if ckp_N == N_u:
        y_ifft = u
    else:  # Apply spectral coarsening
        # FFT
        y_fft = jnp.fft.fftn(u, axes=(3, 2, 1))  # (3, N, N, N)

        # Cut high frequencies. (3, N, N, N) -> (3, ckp_N, ckp_N, ckp_N)
        sl = slice(N_u // 2 - ckp_N // 2, N_u // 2 + ckp_N // 2)
        y_fft_sub = jnp.fft.fftshift(y_fft, axes=(1, 2, 3))[:, sl, sl, sl]
        y_fft_sub = jnp.fft.ifftshift(y_fft_sub, axes=(1, 2, 3))

        # IFFT
        y_ifft = jnp.fft.ifftn(y_fft_sub, axes=(3, 2, 1)) / (N_u / ckp_N) ** 3
        y_ifft = y_ifft.real  # (3, ckp_N, ckp_N, ckp_N)

    # Save to disk
    os.makedirs(dst_path, exist_ok=True)
    file_path = os.path.join(dst_path, f"u_{ckp_N}_{tstep:05d}.npy")
    np.save(file_path, y_ifft)
