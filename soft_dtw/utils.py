import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def pad_inf(inp, before, after):
    o = jnp.pad(inp, (before, after), constant_values=jnp.inf)
    return o


def squared_euclidean_cost(X, Y,
                           return_all: bool=False,
                           log=False):
    def _C(C):
        if log:
            return C + jnp.log(2 - jnp.exp(-C))
        else:
            return C

    X_sqnorms = 0.5 * jnp.sum(X ** 2, axis=1)
    Y_sqnorms = 0.5 * jnp.sum(Y ** 2, axis=1)
    XY = jnp.dot(X, Y.T).astype(X_sqnorms.dtype)

    if return_all:
        C_XY = -XY
        C_XY += X_sqnorms[:, jnp.newaxis]
        C_XY += Y_sqnorms

        C_XX = -jnp.dot(X, X.T)
        C_XX += X_sqnorms[:, jnp.newaxis]
        C_XX += X_sqnorms

        C_YY = -jnp.dot(Y, Y.T)
        C_YY += Y_sqnorms[:, jnp.newaxis]
        C_YY += Y_sqnorms

        return _C(C_XY), _C(C_XX), _C(C_YY)

    else:
        C = -XY
        C += X_sqnorms[:, jnp.newaxis]
        C += Y_sqnorms
        return _C(C)


def get_softmin(gamma, custom_grad=True):
    def softmin_raw(array):
        return -gamma * logsumexp(array / -gamma, axis=-1)

    if not custom_grad:
        return softmin_raw

    softmin = jax.custom_vjp(softmin_raw)

    def softmin_fwd(array):
        return softmin(array), (array / -gamma,)

    def softmin_bwd(res, g):
        scaled_array, = res
        grad = jnp.where(jnp.isinf(scaled_array),
                         jnp.zeros(scaled_array.shape),
                         jax.nn.softmax(scaled_array) * jnp.expand_dims(g, 1)
                         )
        return grad,

    softmin.defvjp(softmin_fwd, softmin_bwd)
    return softmin


def dtw_min_func(args):
    return jnp.min(args, axis=-1)
