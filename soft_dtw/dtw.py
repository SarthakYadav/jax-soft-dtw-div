"""
Implementation of Soft-DTW [1] and Soft-DTW-Divergence [2] in Jax

References
---------
[1] M. Cuturi, M. Blondel. *'Soft-DTW: a Differentiable Loss Function for Time-Series'*, Proc. ICML, 2017.
    [arxiv](https://arxiv.org/abs/1703.01541)

[2] M. Blondel, A. Mensch, JP Vert, *"Differentiable Divergences between Time Series"*, Proc. AISTATS, 2021.
    [arxiv](http://proceedings.mlr.press/v130/blondel21a/blondel21a.pdf)

Based on the following implementations
- https://github.com/google-research/soft-dtw-divergences
- https://github.com/khdlr/softdtw_jax

Hacked together/written by Sarthak Yadav, 2022
"""
import jax

import functools
import numpy as np
import jax.numpy as jnp
from . import utils


def apply_dtw(D, minimum_func=None):
    """

    Args:
        D (jnp.array): squared euclidean cost between input and output
            i.e. squared_euclidean_cost(X,Y)
        minimum_func (Callable, optional): the minimum function to use
            Default: ``None`` will use `dtw_min_func`, which is the standard DTW
            if softmin(), computes Soft-DTW

    Returns:
        jnp.Tensor: dtw value
    """
    if minimum_func is None:
        minimum_func = utils.dtw_min_func
    if D.shape[0] < D.shape[1]:
        D = D.T
    H, W = D.shape

    rows = []
    for row in range(H):
        rows.append(utils.pad_inf(D[row], row, H - row - 1))

    model_matrix = jnp.stack(rows, axis=1)

    init = (
        utils.pad_inf(model_matrix[0], 1, 0),
        utils.pad_inf(model_matrix[1] + model_matrix[0, 0], 1, 0)
    )

    def scan_step(carry, current_antidiagonal):
        two_ago, one_ago = carry

        diagonal = two_ago[:-1]
        right = one_ago[:-1]
        down = one_ago[1:]
        best = minimum_func(jnp.stack([diagonal, right, down], axis=-1))

        next_row = best + current_antidiagonal
        next_row = utils.pad_inf(next_row, 1, 0)
        return (one_ago, next_row), next_row

    carry, ys = jax.lax.scan(scan_step, init, model_matrix[2:], unroll=4)
    return carry[1][-1]


def soft_dtw(X, Y, gamma=1.0):
    dtw_func = functools.partial(apply_dtw, minimum_func=utils.get_softmin(gamma))
    C = utils.squared_euclidean_cost(X, Y)
    return dtw_func(C)


def soft_dtw_divergence(X, Y,
                        gamma=1.0):
    """
    Computes divergence between X and Y

    Args:
         X (jnp.array): signal tensor of shape (N_x, dim)
         Y (jnp.array): (target) signal tensor of shape (N_y, dim)
         gamma (float, optional): gamma value for SoftDTW, default = 1.0
    Returns:
        jnp.Tensor: divergence value
    """
    # if dtw_func is None:
    #     dtw_func = functools.partial(apply_dtw, minimum_func=dtw_min_func)
    dtw_func = functools.partial(apply_dtw, minimum_func=utils.get_softmin(gamma))

    C_XY, C_XX, C_YY = utils.squared_euclidean_cost(X, Y, return_all=True)
    value = dtw_func(C_XY)
    value -= 0.5 * dtw_func(C_XX)
    value -= 0.5 * dtw_func(C_YY)
    return value
