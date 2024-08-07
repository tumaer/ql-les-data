import os
from time import time

import jax
import jax.numpy as jnp
import numpy as np
from jax_sph.io_state import write_h5, write_vtk
from jax_sph.utils import pos_init_cartesian_3d


def get_ckps_list(files_root):
    files = os.listdir(files_root)
    files = [f for f in files if (".npy" in f)]
    files = sorted(files)
    return files


def shift_fn(r, dr, box_size=2 * np.pi):
    """Shift in a periodic box."""
    return (r + dr) % box_size


def spectral_interpolator_wrapper(N, splits=128):
    k = np.fft.fftshift(np.fft.fftfreq(N, 1.0 / N))
    k_field = np.array(np.meshgrid(k, k, k, indexing="ij"), dtype=int)  # (3, N, N, N)

    assert splits & (splits - 1) == 0, "Splits must be a power of 2"

    def interpolate(u, r):
        # u.shape = (3, N, N, N), r.shape - (N^3, 3)

        u_hat = jnp.fft.fftn(u, axes=(3, 2, 1))  # (3, N, N, N)

        def dft(carry, x_i):
            # 2 pi / L = 1
            exponent = jnp.exp(1j * (k_field * x_i[:, None, None, None]).sum(axis=0))
            res = jnp.real(jnp.mean(u_hat * exponent, axis=(1, 2, 3)))
            return None, res

        ### Version 1: Runtime on N=32 with vs without jit: 6s vs 35s
        # u_r = np.zeros_like(r)
        # for i, r_i in enumerate(r):
        #     u_r[i] = dft(None, r_i)[1]

        ### Version 2: Runtime on N=32: 1.35s
        # u_r = jax.lax.scan(dft, None, r)[1]

        ### Version 3: Runtime on N=32 and splits = 4...128: 0.23...0.26s linearly
        def body(carry, x_i):
            u_rs = jax.vmap(dft, in_axes=(None, 0))(None, x_i)[1]
            return None, u_rs

        r_splits = jnp.array(jnp.array_split(r, splits, axis=0))
        u_r = jax.lax.scan(body, None, r_splits)[1]
        u_r = jnp.concatenate(u_r, axis=0)

        return u_r

    return interpolate


def integrate(src_path, dst_path, N=32, dt=0.01, splits=8):
    """Integrate SPH particles along prescribed velocity field.

    Args:
        src_path (str): Path to the directory with the checkpoints. (Source path)
        dst_path (str): Path to the directory where the integrated files will be saved.
        N (int): Number of particles in each dimension.
        dt (float): Time step.
        splits (int): Into how many parts to split 'r' before vmap-ing.
    """
    ckp_path = os.path.join(src_path, "ckp")
    files = get_ckps_list(ckp_path)

    int_path = os.path.join(dst_path, "int")
    os.makedirs(int_path, exist_ok=True)

    r = pos_init_cartesian_3d(2 * np.pi * np.ones(3), 2 * np.pi / N)
    write_h5({"r": r, "u": r * 0.0}, os.path.join(int_path, "step_00000.h5"))
    write_vtk({"r": r, "u": r * 0.0}, os.path.join(int_path, "step_00000.vtk"))

    interpolator = spectral_interpolator_wrapper(N, splits=8)
    t0 = time()
    t_int = 0.0
    for i, file in enumerate(files):
        u = np.load(os.path.join(ckp_path, file))

        t_temp = time()
        u_r = interpolator(u, r)
        t_int += time() - t_temp

        r = shift_fn(r, dt * u_r)
        write_h5({"r": r, "u": u_r}, os.path.join(int_path, f"step_{i+1:05d}.h5"))
        write_vtk({"r": r, "u": u_r}, os.path.join(int_path, f"step_{i+1:05d}.vtk"))

    t_tot = time() - t0
    print(f"t_tot = {t_tot:.3f}, t_int = {t_int:.3f}")
