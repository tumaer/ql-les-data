import os

import finufft
import jax
import jax.numpy as jnp
import numpy as np
from jax_sph.io_state import read_h5
from jax_sph.utils import pos_init_cartesian_2d, pos_init_cartesian_3d

from l3es.relax import relax_wrapper
from l3es.utils import rho_computer


def get_ckps_list(files_root):
    files = os.listdir(files_root)
    files = [f for f in files if (".npy" in f)]
    files = sorted(files)
    return files


def shift_fn(r, dr, box_size=2 * np.pi):
    """Shift in a periodic box."""
    return (r + dr) % box_size


def spectral_interpolator_wrapper(N, fft_axes=(1, 2, 3), splits=128, backend="nufft"):
    is_3d = len(fft_axes) == 3
    k = np.fft.fftshift(np.fft.fftfreq(N, 1.0 / N))
    k_tuple = (k, k, k) if is_3d else (k, k)
    k_field = np.array(np.meshgrid(*k_tuple, indexing="ij"), dtype=int)  # (3, N, N, N)
    norm = N ** len(fft_axes)

    assert splits & (splits - 1) == 0, "Splits must be a power of 2"
    assert backend in ["dft", "nufft"]

    def interpolate_finufft(u, r):
        # u.shape = (3, N, N, N) or (2, N, N), r.shape = (num_particles, dim)
        u_hat = np.fft.fftshift(
            np.fft.fftn(np.asarray(u), axes=fft_axes), axes=fft_axes
        )

        # 1. Make the entire batched coefficient array contiguous ONCE
        # FINUFFT expects shape (n_trans, N, N, N) for multi-transforms
        coeff = np.ascontiguousarray(u_hat, dtype=np.complex128)

        # Extract and format coordinates
        x = np.ascontiguousarray(r[:, 0], dtype=np.float64)
        y = np.ascontiguousarray(r[:, 1], dtype=np.float64)
        if is_3d:
            z = np.ascontiguousarray(r[:, 2], dtype=np.float64)
            # 2. Vectorized call: passing the (3, N, N, N) coeff array directly
            # Returns vals of shape (3, num_particles)
            vals = finufft.nufft3d2(x, y, z, coeff, isign=1)
        else:
            vals = finufft.nufft2d2(x, y, coeff, isign=1)

        # 3. Transpose the result to match (num_particles, channels) and normalize
        u_r = np.real(vals).T / norm

        return jnp.asarray(u_r)

    def interpolate_dft(u, r):
        # u.shape = (3, N, N, N) or (2, N, N), r.shape - (N^3, 3)

        u_hat = jnp.fft.fftn(u, axes=fft_axes)  # (3, N, N, N)
        u_hat = jnp.fft.fftshift(u_hat, axes=fft_axes)

        def idft(carry, x_i):
            # 2 pi / L = 1
            exponent = jnp.exp(1j * ((k_field.T * x_i).T).sum(axis=0))
            res = jnp.real(jnp.mean(u_hat * exponent, axis=fft_axes))
            return None, res

        ### Version 1: Runtime on N=32 with vs without jit: 6s vs 35s
        # u_r = np.zeros_like(r)
        # for i, r_i in enumerate(r):
        #     u_r[i] = idft(None, r_i)[1]

        ### Version 2: Runtime on N=32: 1.35s
        # u_r = jax.lax.scan(idft, None, r)[1]

        ### Version 3: Runtime on N=32 and splits = 4...128: 0.23...0.26s linearly
        def body(carry, x_i):
            u_rs = jax.vmap(idft, in_axes=(None, 0))(None, x_i)[1]
            return None, u_rs

        r_splits = jnp.array(jnp.array_split(r, splits, axis=0))
        u_r = jax.lax.scan(body, None, r_splits)[1]
        u_r = jnp.concatenate(u_r, axis=0)

        return u_r

    return interpolate_dft if backend == "dft" else interpolate_finufft


def set_up_integrator(state_0_path, dim, N, dft_splits, u_ref, interp_backend):
    L = 2 * np.pi
    fft_axes = (1, 2, 3) if dim == 3 else (1, 2)

    if state_0_path is not None:
        r = read_h5(state_0_path)["r"]
    else:
        if dim == 3:
            r = pos_init_cartesian_3d(L * np.ones(3), L / N)
        else:
            r = pos_init_cartesian_2d(L * np.ones(2), L / N)

    comp_rho = rho_computer(N, dim=dim, L=L)
    interpolator = spectral_interpolator_wrapper(
        N, fft_axes, splits=dft_splits, backend=interp_backend
    )
    relax_fn, sph_fn = relax_wrapper(N, dim, L, is_physical=True, u_ref=u_ref)

    return r, comp_rho, interpolator, relax_fn, sph_fn, L, fft_axes
