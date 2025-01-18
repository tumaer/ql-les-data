"""Turbulence jax-sph utils."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax.numpy as jnp
import numpy as np
from jax import lax, ops, vmap
from jax.scipy.special import factorial
from jax_sph.io_state import read_h5
from jax_sph.jax_md import space
from jax_sph.kernel import QuinticKernel
from jax_sph.utils import pos_init_cartesian_2d, pos_init_cartesian_3d
from numpy import array
from scipy.spatial import KDTree

from l3es.visualize import plot_e_k, plot_views

EPS = jnp.finfo(float).eps


class M4PrimeKernel:
    """The M'4 kernel, based on "Analysis of interpolation schemes for the accurate
    estimation of energy spectrum in Lagrangian methods", Shi et al., 2013"""

    def __init__(self, h, dim=3):
        self._one_over_h = 1.0 / h
        self._normalized_cutoff = 2.0
        self.cutoff = self._normalized_cutoff * h
        self.pnorm = jnp.inf

    def w(self, r):
        """Evaluates the kernel at the radial displacement vector r."""
        q = jnp.abs(r) * self._one_over_h
        q1 = 1 - 2.5 * q**2 + 1.5 * q**3
        q2 = 0.5 * (1 - q) * (2 - q) ** 2

        res = jnp.where(q < 1, q1, 0)
        res = jnp.where((q >= 1) * (q < 2), q2, res)
        return jnp.prod(res)  # , axis=1


def pbc_copy_scalar(
    r: array, f: array, box_size: array, halo: float, dim: int, unsorted: bool = True
):
    """Copy particles of a scalar field for PBCs on a rectangular domain

    Args:
        r (np.ndarray): coordinates of N particles of shape (N, dim)
        f (np.ndarray): vector field, e.g. velocity of shape (N, dim)
        box_size (np.ndarray): Domain box of form np.array([x_size, y_size, z_size])
        halo (float): Width of halo region, e.g. 3h for Quintic spline
        dim (int): dimension of the data
        unsorted (bool): whether indices remain the same

    Returns:
        (np.ndarray, np.ndarray): new positions and properties after copying
    """
    if dim == 1:
        # get right side indices
        right = np.where(r <= halo, True, False)

        # get left side indices
        left = np.where(r >= box_size - halo, True, False)

        # concatenate pbc values
        r = np.concatenate((r, r[right] + box_size, r[left] - box_size))
        f = np.concatenate((f, f[right], f[left]))

        # rid of possible overlap
        ind = np.unique(r, axis=0, return_index=True)[1]
        ind = sorted(ind) if unsorted else ind
        r_pbc = r[ind][:, None]
        f_pbc = f[ind]

    elif dim == 2:
        # left and right side first
        # get indices
        right = np.where(r[:, 0] <= halo, True, False)
        left = np.where(r[:, 0] >= box_size[0] - halo, True, False)

        # concatenate pbc values
        dr = np.array([box_size[0], 0.0])[None, :]
        r = np.concatenate((r, r[right, :] + dr, r[left, :] - dr), axis=0)
        f = np.concatenate((f, f[right], f[left]))

        # now top and bottom
        # get indices
        top = np.where(r[:, 1] <= halo, True, False)
        bottom = np.where(r[:, 1] >= box_size[1] - halo, True, False)

        # concatenate pbc values
        dr = np.array([0.0, box_size[1]])[None, :]
        r = np.concatenate((r, r[top, :] + dr, r[bottom, :] - dr), axis=0)
        f = np.concatenate((f, f[top], f[bottom]))

        # rid of possible overlap
        ind = np.unique(r, axis=0, return_index=True)[1]
        ind = sorted(ind) if unsorted else ind
        r_pbc = r[ind, :]
        f_pbc = f[ind]

    elif dim == 3:
        # left and right side first
        # get indices
        right = np.where(r[:, 0] <= halo, True, False)
        left = np.where(r[:, 0] >= box_size[0] - halo, True, False)

        # concatenate pbc values
        dr = np.array([box_size[0], 0.0, 0.0])[None, :]
        r = np.concatenate((r, r[right, :] + dr, r[left, :] - dr), axis=0)
        f = np.concatenate((f, f[right], f[left]))

        # now top and bottom
        # get indices
        top = np.where(r[:, 1] <= halo, True, False)
        bottom = np.where(r[:, 1] >= box_size[1] - halo, True, False)

        # concatenate pbc values
        dr = np.array([0.0, box_size[1], 0.0])[None, :]
        r = np.concatenate((r, r[top, :] + dr, r[bottom, :] - dr), axis=0)
        f = np.concatenate((f, f[top], f[bottom]))

        # now front and back
        # get indices
        back = np.where(r[:, 2] <= halo, True, False)
        front = np.where(r[:, 2] >= box_size[2] - halo, True, False)

        # concatenate pbc values
        dr = np.array([0.0, 0.0, box_size[2]])[None, :]
        r = np.concatenate((r, r[back, :] + dr, r[front, :] - dr), axis=0)
        f = np.concatenate((f, f[back], f[front]))

        # rid of possible overlap
        ind = np.unique(r, axis=0, return_index=True)[1]
        ind = sorted(ind) if unsorted else ind
        r_pbc = r[ind, :]
        f_pbc = f[ind]

    return r_pbc, f_pbc


def mls_2nd_order(
    r, r_target, f, box_size, dx, dim, kernel_name="M4Prime", h_factor=None
):
    """2nd-order moving least squares interpolation for periodic flows in a
    rectangular box.

    Based on, "Analysis of interpolation schemes for the accurate estimation of
    energy spectrum in Lagrangian methods", Shi et al., 2013

    Args:
        r (np.ndarray): coordinates of N particles of shape (N, dim)
        r_target (np.ndarray): coordinates of target particles of shape (N, dim)
        f (np.ndarray): scalar field, e.g. velocity component of shape (N,)
        box_size (np.ndarray): Domain box, e.g. np.array([1., 2., 3.])
        dx (float): average particle spacing
        dim (int): dimension of the vector field

    Returns:
        (np.ndarray): interpolated values of f on r_target
    """

    # define kernel function
    if kernel_name == "Quintic":
        h_factor = 2 / 3 if h_factor is None else h_factor
        kernel_fn = QuinticKernel(h=h_factor * dx, dim=dim)
        kernel_fn.pnorm = 2
    elif kernel_name == "M4Prime":
        h_factor = 0.85 if h_factor is None else h_factor
        kernel_fn = M4PrimeKernel(h=h_factor * dx, dim=dim)
        kernel_fn.pnorm = np.inf
    else:
        raise NotImplementedError(f"Kernel {kernel_name} not implemented.")

    # displacement function for neighbors list
    displacement_fn, shift_fn = space.periodic(side=box_size)

    # number of target particles
    n_target = jnp.shape(r_target)[0]

    # enforce periodic boundary conditions
    r_pbc, f_pbc = pbc_copy_scalar(r, f, box_size, kernel_fn.cutoff, dim)

    # compute edge list
    tree = KDTree(r_pbc)
    senders = tree.query_ball_point(r_target, kernel_fn.cutoff, p=kernel_fn.pnorm)
    i_s = np.repeat(range(n_target), [len(x) for x in senders])
    j_s = np.concatenate(senders, axis=0)

    # precompute quantities
    r_ji = vmap(displacement_fn)(r_pbc[j_s], r_target[i_s])
    if kernel_name == "M4Prime":
        w_dist = vmap(kernel_fn.w)(r_ji)
    elif kernel_name == "Quintic":
        rel_distances = np.linalg.norm(r_ji, axis=1, ord=2)
        w_dist = kernel_fn.w(rel_distances)
    else:
        raise NotImplementedError(f"Kernel {kernel_name} not implemented.")

    # define size of the linear system of equations
    sum1 = factorial(dim) / factorial(dim - 1)
    sum2 = factorial(dim + 1) / (factorial(dim - 1) * 2)
    mat_size = round(1 + sum1 + sum2)  # 3/6/10 for dim=1/2/3

    # calculate indices
    ind_d = jnp.diag_indices(dim)
    ind_u = jnp.triu_indices(dim, 1)

    # mls matrix entries
    def matrix(w_dist, r_ji):
        tensor = jnp.tensordot(r_ji, r_ji, axes=0)

        row = jnp.ones(mat_size)
        row = row.at[1 : dim + 1].mul(r_ji)
        row = row.at[dim + 1 : 2 * dim + 1].mul(tensor[ind_d] * 0.5)
        row = row.at[2 * dim + 1 :].mul(tensor[ind_u])

        column = jnp.ones(mat_size)
        column = column.at[1 : dim + 1].mul(r_ji)
        column = column.at[dim + 1 : 2 * dim + 1].mul(tensor[ind_d])
        column = column.at[2 * dim + 1 :].mul(tensor[ind_u] * 2)

        return jnp.tensordot(column, row, axes=0) * w_dist  # (3x3), (6x6), (10x10)

    # calculate matrix
    temp = vmap(matrix)(w_dist, r_ji)
    mat = ops.segment_sum(temp, i_s, n_target)

    # define solution vector entries
    def vector(w_dist, r_ji, f_j):
        tensor = jnp.tensordot(r_ji, r_ji, axes=0)
        vector = jnp.ones(mat_size) * w_dist * f_j
        vector = vector.at[1 : dim + 1].mul(r_ji)
        vector = vector.at[dim + 1 : 2 * dim + 1].mul(tensor[ind_d])
        vector = vector.at[2 * dim + 1 :].mul(tensor[ind_u] * 2)
        return vector

    # calculate vector
    temp = vmap(vector)(w_dist, r_ji, f_pbc[j_s])
    vec = ops.segment_sum(temp, i_s, n_target)

    # function for solving the system
    def solve_lin(matrix, vector):
        temp = jnp.linalg.solve(matrix, vector)
        return temp[0]

    # calculate interpolated value
    f_target = vmap(solve_lin)(mat, vec)

    return f_target


def ur_to_u_mls_wrapper(N, L, dim=3):
    """Interpolate velocities from particles to regular grid using MLS.

    Args:
        N (int): Number of particles per direction.
        L (float): Domain size.
        dim (int, optional): Dimension. For now only 3.

    Returns:
        function: Interpolation function.
    """

    dx = L / N
    box_size = L * np.ones(dim)
    if dim == 3:
        r_grid = pos_init_cartesian_3d(box_size, dx)
    else:
        r_grid = pos_init_cartesian_2d(box_size, dx)

    def body(r, u_r):
        """Core interpolate function.

        Args:
            r (np.ndarray): particle positions of shape (N^dim, dim)
            u_r (np.ndarray): particle velocities of shape (N^3, dim)

        Returns:
            u (np.ndarray): interpolated velocities on regular grid of shape (3,N,N,N)
        """
        assert (r.shape == u_r.shape) and (r.shape[0] == N**dim)

        u_grid = [
            mls_2nd_order(
                r, r_grid, u_r[:, i], box_size, dx, dim, kernel_name="Quintic"
            )
            for i in range(dim)
        ]

        return jnp.array(u_grid).reshape(dim, -1).T

    return body


def ur_to_u_dft_wrapper(N, L, dim=3, fft_axes=(1, 2, 3)):
    """Interpolate velocities from particles to regular grid using DFT.

    Args:
        N (int): Number of particles per direction.
        L (float): Domain size.
        dim (int, optional): Dimension. For now only 3.

    Returns:
        function: Interpolation function.
    """

    k = jnp.fft.fftfreq(N, 1.0 / N)
    k_tuple = (k, k, k) if dim == 3 else (k, k)
    k_field = jnp.array(jnp.meshgrid(*k_tuple, indexing="ij"), dtype=int)  # (3,N,N,N)

    def body(r, u_r):
        """Core interpolate function.

        Args:
            r (np.ndarray): particle positions of shape (N^dim, dim)
            u_r (np.ndarray): particle velocities of shape (N^3, dim)

        Returns:
            u (np.ndarray): interpolated velocities on regular grid of shape (3,N,N,N)
        """

        def dft(carry, k_i):
            exponent = jnp.exp(-1j * (k_i * r).sum(axis=1))  # (N^3,)
            res = jnp.sum(u_r.T * exponent, axis=1)  # (3,)
            return None, res

        u_hat = lax.scan(dft, None, (k_field.T).reshape(-1, dim))[1]  # (N^3, 3)
        u_hat = (u_hat.reshape(*k_field.shape[::-1])).T  # (3, N, N, N)

        u_grid = jnp.fft.ifftn(u_hat, axes=fft_axes)  # (3, N, N, N)
        u_grid = u_grid.real

        return u_grid.reshape(dim, -1).T

    return body


if __name__ == "__main__":
    for step in [100, 1000]:
        N = 32
        L = 2 * np.pi
        dx = L / N
        dim = 3
        u_ref = 4.0
        data_path = "results_hit_192_3"
        vis_path = os.path.join(data_path, "int_vis_")

        # load h5 frame
        state = read_h5(os.path.join(data_path, f"int/step_{step:05d}.h5"))
        r, u_r = state["r"], state["u"]
        # plot_views((r.T).reshape(3,N,N,N), (u_r.T).reshape(3,N,N,N), L/N, step,
        #     save_path=vis_path, u_ref=4.0)

        # interpolate velocities to regular grid
        is_mls = False
        if is_mls:
            r_grid, u_grid = ur_to_u_mls_wrapper(N, L, dim)(r, u_r)  # (N^3,3), (N^3,3)
        else:
            r_grid = pos_init_cartesian_3d(L * np.ones(3), L / N)
            # r_eval=(r-dx/2)%L  # iFFT gives the values from (0,0,0), not dx/2*(1,1,1)
            u_grid = ur_to_u_dft_wrapper(N, L)(r, u_r)

        r_grid = np.asarray((r_grid.T).reshape(3, N, N, N))
        u_grid = np.asarray((u_grid.T).reshape(3, N, N, N))

        # visualize
        plot_e_k(u_grid, step, save_path=vis_path, dim=dim)
        plot_views(r_grid, u_grid, L / N, step, save_path=vis_path, u_ref=u_ref)
        print("Done.")

        # nohup python main.py config=configs/hit_192.yaml mode=integrate \
        #     int.dst_path=results_hit_192_2/ gpu=2 >> hit_192_5_0002_2.out 2>&1 &
