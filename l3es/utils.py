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


def energy_spectrum(vel, mul_fac: float = 1.0, is_scalar_field: bool = False, dim=3):
    """JAX implemented energy spectrum computation on a grid.

    Adapted from JAX-FLUIDS 1.0 implementation."""

    if dim == 2:
        vel = vel[:2, :, :, 0]
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
    vel_hat /= n**dim

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


def spectral_filtering(u, ckp_N):
    """Spectral filtering for LES reference data.

    Args:
        u (jnp.ndarray): Flow field with shape (3, N, N, N) or (2, N, N).
        ckp_N (int): How many spatial modes to keep (after spectral filtering).

    Returns:
        jnp.ndarray: Flow field of shape (3, ckp_N, ckp_N, ckp_N) or (2, ckp_N, ckp_N).
    """

    N_u = u.shape[1]
    dim = len(u)
    fft_axes = (1, 2, 3) if dim == 3 else (1, 2)

    # FFT
    y_fft = jnp.fft.fftn(u, axes=fft_axes)  # (3, N, N, N)  TODO: this was (3,2,1)

    # Cut high frequencies. (3, N, N, N) -> (3, ckp_N, ckp_N, ckp_N)
    sl = slice(N_u // 2 - ckp_N // 2, N_u // 2 + ckp_N // 2)
    slices = tuple([slice(None)] + [sl] * dim)
    y_fft_sub = jnp.fft.fftshift(y_fft, axes=fft_axes)[slices]
    y_fft_sub = jnp.fft.ifftshift(y_fft_sub, axes=fft_axes)

    # IFFT
    # TODO: this was (3,2,1)
    y_ifft = jnp.fft.ifftn(y_fft_sub, axes=fft_axes) / (N_u / ckp_N) ** dim
    y_ifft = y_ifft.real  # (3, ckp_N, ckp_N, ckp_N)

    # # Orientation changes during spectral filtering from DNS to LES grid
    # import matplotlib.pyplot as plt
    # _, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].imshow(u[0])
    # axs[1].imshow(y_ifft[0])
    # plt.savefig("orientation_check.png")

    return y_ifft


def write_u(u, tstep, dst_path, ckp_N):
    """Write flow field to disk.

    Args:
        u (jnp.ndarray): Flow field with shape (3, N, N, N) or (3, N, N, 1).
        tstep (int): Current time step.
        dst_path (str): Where to write results.
        ckp_N (int): How many spatial modes to keep (after spectral filtering).

    Writes:
        u_{ckp_N}_{tstep:05d}.npy: Flow field of shape (3, N_sub, N_sub, N_sub) or
            (2, N_sub, N_sub)
    """

    N_u = u.shape[1]
    dim = 2 if u.shape[-1] == 1 else 3
    u = u if dim == 3 else u[:2, :, :, 0]
    assert ckp_N <= N_u, "ckp_N must be less than or equal to N."
    assert ckp_N % 2 == 0 and N_u % 2 == 0, "Only tested for even N and ckp_N."

    y_ifft = u if ckp_N == N_u else spectral_filtering(u, ckp_N)

    # Save to disk
    os.makedirs(dst_path, exist_ok=True)
    file_path = os.path.join(dst_path, f"u_{ckp_N}_{tstep:05d}.npy")
    np.save(file_path, y_ifft)
