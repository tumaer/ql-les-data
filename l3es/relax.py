"""Utilities for SPH particle relaxation. See Neural SPH (Toshev et al., 2024)."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, ops, vmap
from jax_sph.eos import TaitEoS
from jax_sph.jax_md import space
from jax_sph.kernel import QuinticKernel
from jax_sph.partition import neighbor_list
from jax_sph.utils import pos_init_cartesian_2d, pos_init_cartesian_3d

EPS = jnp.finfo(float).eps


def relax_wrapper(N, dim=3, L=2 * np.pi, is_physical=False, u_ref=None):
    """Compute pressure gradient term from NSE.

    Args:
        N (int): Number of particles per dimension.
        dim (int): Dimension.
        L (float): Box size.
        is_physical (bool): Whether to use physical units internally. Doesn't affect
            the output.
        u_ref (float): Reference velocity for equation of state.

    Returns:
        Callable which takes coordinates `r` and returns density `rho`.
    """

    N_tot = N**dim
    dx_phys = L / N
    l_ref = 1.0 if is_physical else dx_phys
    dx = dx_phys / l_ref  # non-dimensionalize

    rho_ref = 1.0  # reference density
    mass = dx**dim * rho_ref

    box_size = np.ones(dim) * L / l_ref
    displacement_fn, shift_fn = space.periodic(side=box_size)
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
    neighbor_fn.update
    nbrs = neighbor_fn.allocate(pos_demo, num_particles=N_tot)
    print("nbrs with cell capacity:", nbrs.cell_list_capacity)
    nbrs_update = jit(nbrs.update)

    u_eos = u_ref / l_ref
    c_eos = 10 * u_eos  # speed of sound
    p_eos = c_eos**2 * rho_ref  # reference pressure
    print(f"Reference pressure: {p_eos:.4f}")
    eos = TaitEoS(p_ref=p_eos, rho_ref=rho_ref, p_background=0.0, gamma=1.0)

    def normalize_length(r):
        return r / l_ref

    def denormalize_length(r):
        return r * l_ref

    @jax.jit
    def loop_body(r):
        r = normalize_length(r)

        nbrs = nbrs_update(r, num_particles=N_tot)
        i_s, j_s = nbrs.idx
        r_i_s, r_j_s = r[i_s], r[j_s]
        dr_i_j = vmap(displacement_fn)(r_i_s, r_j_s)
        dist = space.distance(dr_i_j)
        w_dist = vmap(kernel_fn.w)(dist)

        rho = mass * ops.segment_sum(w_dist, i_s, N_tot)
        p = vmap(eos.p_fn)(rho)

        def acceleration_fn(r_ij, d_ij, rho_i, rho_j, p_i, p_j):
            # Compute unit vector, above eq. (6), Zhang (2017). Sign flipped here.
            e_ij = r_ij / (d_ij + EPS)

            # Compute kernel gradient
            kernel_der = kernel_fn.grad_w(d_ij)
            kernel_grad = kernel_der * e_ij

            # Compute density-weighted pressure (weighted arithmetic mean)
            p_ij = (rho_j * p_i + rho_i * p_j) / (rho_i + rho_j)

            # Eq. (8), Adami (2012) with constant `mass`
            prefactor = mass * ((1 / rho_i) ** 2 + (1 / rho_j) ** 2)
            acc = -prefactor * p_ij * kernel_grad

            # Add transport velocity acceleration term on top (Eq. 13)
            # 0.5 comes from the integration scheme
            acc += 0.5 * prefactor * (-p_eos) * kernel_grad

            return acc

        out = vmap(acceleration_fn)(
            dr_i_j,
            dist,
            rho[i_s],
            rho[j_s],
            p[i_s],
            p[j_s],
        )
        acc = ops.segment_sum(out, i_s, N_tot)
        acc = denormalize_length(acc)
        return acc

    return loop_body
