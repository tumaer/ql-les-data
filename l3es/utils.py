"""Utilities for L3ES."""

import os

import jax.numpy as jnp
import jax_cfd.base as cfd  # version 0.2.1
import numpy as np
from jax import jit, ops, random, vmap
from jax_cfd.collocated import initial_conditions, pressure
from jax_sph.jax_md import space
from jax_sph.kernel import QuinticKernel
from jax_sph.partition import neighbor_list
from jax_sph.utils import pos_init_cartesian_2d, pos_init_cartesian_3d

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

    Args:
        vel (jnp.ndarray): Velocity field with shape (3, N, N, N) or (3, N, N, 1).
        mul_fac (float): Multiplication factor for energy spectrum.
        is_scalar_field (bool): Whether the field is scalar or vector.
        dim (int): Dimension of the field.

    Code based on JAX-FLUIDS implementation."""

    if dim == 2:  # applies when the physics is 2D but data is padded to 3D
        vel = vel[:2, :, :, 0]
    ns = vel.shape[1:]

    # check for square box with equal side length
    assert jnp.array_equal(ns, jnp.ones(dim) * ns[0])

    # common resolution
    n = ns[0]

    # Fourier transform
    if dim == 1:
        # TODO: check whether 1D is working
        vel_hat = jnp.fft.rfftn(vel)
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
    # axs[0].set_title(f"[{u[0].min():.2f}, {u[0].max():.2f}]")
    # axs[1].imshow(y_ifft[0])
    # axs[1].set_title(f"[{y_ifft[0].min():.2f}, {y_ifft[0].max():.2f}]")
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


def rho_computer(N, dim=3, L=2 * np.pi):
    """Compute density from coordinates or from a rollout file.

    Args:
        N (int): Number of particles per dimension.
        dim (int): Dimension.
        L (float): Box size.

    Returns:
        Callable which takes coordinates `r` and returns density `rho`.
    """

    N_tot = N**dim
    dx = L / N
    mass = dx**dim
    box_size = np.ones(dim) * L
    displacement_fn, _ = space.periodic(side=box_size)
    kernel_fn = QuinticKernel(h=dx, dim=dim)

    # set up neighbor search routine
    if dim == 3:
        pos_demo = pos_init_cartesian_3d(box_size, dx)
    else:
        pos_demo = pos_init_cartesian_2d(box_size, dx)
    neighbor_fn = neighbor_list(
        displacement_fn,
        box_size,
        backend="jaxmd_vmap",
        r_cutoff=kernel_fn.cutoff,
        capacity_multiplier=2.0,
        mask_self=False,
        num_particles_max=N_tot,
        pbc=[True] * dim,
    )
    nbrs = neighbor_fn.allocate(pos_demo, num_particles=N_tot)
    nbrs_update = jit(nbrs.update)

    def comp_rho(r):
        nbrs = nbrs_update(r, num_particles=N_tot)
        i_s, j_s = nbrs.idx
        r_i_s, r_j_s = r[i_s], r[j_s]
        dr_i_j = vmap(displacement_fn)(r_i_s, r_j_s)
        dist = space.distance(dr_i_j)
        w_dist = vmap(kernel_fn.w)(dist)

        rho = mass * ops.segment_sum(w_dist, i_s, N_tot)  # density summation
        # rho = jnp.where(rho < 0.98, 1, rho)  # detect free surface
        return rho

    return comp_rho


def comp_divergence(u, dx, version=2):
    """Computes divergence of vector field
    u (array) -> vector field components [Fx,Fy,Fz,...]
    dx (float) -> spacing between points in every directions
    """
    if version == 0:
        num_dims = len(u)
        return np.ufunc.reduce(
            np.add, [np.gradient(u[i], dx, axis=i) for i in range(num_dims)]
        )
    elif version == 1:
        dudx = np.gradient(u[0], dx, axis=0)
        dvdy = np.gradient(u[1], dx, axis=1)
        return dudx + dvdy
    elif version == 2:  # this version by construction zeros out the boundaries
        res = np.zeros_like(u[0])
        res[1:-1, 1:-1] = (
            u[0, 2:, 1:-1] - u[0, :-2, 1:-1] + u[1, 1:-1, 2:] - u[1, 1:-1, :-2]
        ) / (2 * dx)
        return res


def comp_divergence_3d(u, dx, version=2):
    """Computes divergence of vector field
    u (array) -> vector field components [Fx,Fy,Fz,...]
    dx (float) -> spacing between points in every directions
    """
    if version == 0:
        num_dims = len(u)
        return np.ufunc.reduce(
            np.add, [np.gradient(u[i], dx, axis=i) for i in range(num_dims)]
        )
    elif version == 1:
        dudx = np.gradient(u[0], dx, axis=0)
        dvdy = np.gradient(u[1], dx, axis=1)
        dwdz = np.gradient(u[2], dx, axis=2)
        return dudx + dvdy + dwdz
    elif version == 2:  # this version by construction zeros out the boundaries
        res = np.zeros_like(u[0])
        res[1:-1, 1:-1, 1:-1] = (
            u[0, 2:, 1:-1, 1:-1]
            - u[0, :-2, 1:-1, 1:-1]
            + u[1, 1:-1, 2:, 1:-1]
            - u[1, 1:-1, :-2, 1:-1]
            + u[2, 1:-1, 1:-1, 2:]
            - u[2, 1:-1, 1:-1, :-2]
        ) / (2 * dx)
        return res


def comp_vorticity(u, dx):
    """Computes vorticity of vector field
    u (array) -> vector field components [Fx,Fy,Fz,...]
    dx (float) -> spacing between points in every directions
    """
    dudy = np.gradient(u[0], dx, axis=1)
    dvdx = np.gradient(u[1], dx, axis=0)
    return dvdx - dudy


def comp_vorticity_3d(u, dx):
    """Computes vorticity of vector field
    u (array) -> vector field components [Fx,Fy,Fz,...]
    dx (float) -> spacing between points in every directions
    """
    dudy = np.gradient(u[0], dx, axis=1)
    dvdx = np.gradient(u[1], dx, axis=0)
    dudz = np.gradient(u[0], dx, axis=2)
    dwdx = np.gradient(u[2], dx, axis=0)
    dvdz = np.gradient(u[1], dx, axis=2)
    dwdy = np.gradient(u[2], dx, axis=1)

    vort_x = dwdy - dvdz
    vort_y = dudz - dwdx
    vort_z = dvdx - dudy

    return np.array([vort_x, vort_y, vort_z])


def comp_dissipation_rate(u_hat, nu):
    """Computes exact physical dissipation directly from the 3D spectral gradients."""
    n = u_hat.shape[1]
    fft_axes = (1, 2, 3)

    # wavenumber grid setup (same as in spectral solver)
    kx = jnp.fft.fftfreq(n, 1.0 / n)
    kz = kx[: (n // 2 + 1)].copy()
    kz = kz.at[-1].set(-1 * kz[-1])
    kkk = jnp.array(jnp.meshgrid(kx, kx, kz, indexing="ij"), dtype=int)

    # spectral curl
    curl = jnp.fft.irfftn(1j * jnp.cross(kkk, u_hat, axis=0), axes=fft_axes)

    # nu * < |omega|^2 >
    return nu * jnp.mean(jnp.sum(curl**2, axis=0))


def make_incompressible_real(u, N, L, target_dim=3):
    """Make a velocity field incompressible.

    Implemented only for 2D so far.
    """
    grid = cfd.grids.Grid((N, N), domain=((0, L), (0, L)))
    v0 = initial_conditions.filtered_velocity_field(
        random.PRNGKey(0), grid, 7, 4, iterations=4
    )

    v0[0].array.data = u[0]
    v0[1].array.data = u[1]
    v0 = pressure.projection(v0)

    v0 = jnp.array([v.data for v in v0])[..., None]  # (2, N, N, 1)

    if target_dim == 3:
        v0 = jnp.concatenate([v0, jnp.zeros((1, N, N, 1))], axis=0)
    return v0


def make_incompressible_spectral(N, fft_axes=(1, 2, 3)):
    """Create a function that projects velocity fields onto the incompressible manifold.

    Very similar to the spectral solver code.
    Tested only on 2D so far.

    Uses spectral method with the projection operator P(k) = I - k⊗k/|k|².
    Based on the spectralDNS implementation structure.
    """
    # Set up wavenumbers like in the original code
    kx = jnp.fft.fftfreq(N, 1.0 / N)
    kx_tuple = (kx,) * 2 if len(fft_axes) == 3 else (kx,)
    kz = kx[: (N // 2 + 1)].copy()
    kz = kz.at[-1].set(-1 * kz[-1])

    # Create wavenumber mesh
    kkk = jnp.array(jnp.meshgrid(*kx_tuple, kz, indexing="ij"), dtype=int)
    if len(fft_axes) == 2:
        kkk = jnp.concatenate([kkk, jnp.zeros((1, N, N // 2 + 1))], axis=0)

    # Compute k²
    kkk2 = jnp.sum(kkk * kkk, 0, dtype=int)

    # Compute k/k² (with handling of k=0 mode)
    kkk_over_kkk2 = kkk.astype(float) / jnp.where(kkk2 == 0, 1, kkk2).astype(float)

    def project_velocity(u=None, u_hat=None):
        """Project velocity field to be incompressible.

        Args:
            u: (3, N, N, 1) Real space velocity field (if provided)
            u_hat: Fourier space velocity field (if provided)

        Returns:
            u_incompressible: Real space incompressible velocity
            u_hat_incompressible: Fourier space incompressible velocity
        """
        if u_hat is None and u is not None:
            # Transform to Fourier space if needed
            u_hat = jnp.fft.rfftn(u, axes=fft_axes).squeeze()
        elif u_hat is None and u is None:
            raise ValueError("Must provide either u or u_hat")

        # Compute the projection in Fourier space
        # First get the divergent component
        p_hat = jnp.sum(u_hat * kkk_over_kkk2, axis=0)
        # Subtract it to get divergence-free field
        u_hat_incompressible = u_hat - p_hat * kkk

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        # def plt_field(ax, v):
        #     ax.imshow(v, vmin=-5, vmax=5)
        #     ax.set_title(f"div(u_hat) [{v.min():.2f}, {v.max():.2f}, {v.std():.2f}]")
        # plt_field(axs[0,0], (u_hat.real*kkk).sum(0))
        # plt_field(axs[0,1], (u_hat.imag*kkk).sum(0))
        # plt_field(axs[1,0], (u_hat_incompressible.real*kkk).sum(0))
        # plt_field(axs[1,1], (u_hat_incompressible.imag*kkk).sum(0))
        # plt.tight_layout()
        # plt.savefig(f"incompr_figure_spectral.png")

        if u is not None:
            # Transform back to real space if input was in real space
            u_incompressible = jnp.fft.irfftn(u_hat_incompressible, axes=fft_axes)
            return u_incompressible, u_hat_incompressible

        return None, u_hat_incompressible

    return project_velocity
