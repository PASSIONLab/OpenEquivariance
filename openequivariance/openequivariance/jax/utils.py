import jax.numpy as jnp
from jax.interpreters import ad


def clean_tensors(*tensors):
    tensors_clean = []
    for t in tensors:
        result = t
        if type(t) is ad.Zero or ad.is_undefined_primal(t):
            result = jnp.zeros(t.aval.shape, t.aval.dtype)
        tensors_clean.append(result)
    return tensors_clean
