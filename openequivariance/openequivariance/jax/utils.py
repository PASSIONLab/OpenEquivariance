import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad


def reorder_jax_helper(schedule, weights_in, direction, has_batch_dim):
    assert direction in ["forward", "backward"]

    specs = schedule.weight_reordering_info(weights_in, has_batch_dim)
    weights_out = jnp.zeros_like(weights_in)

    for spec in specs:
        parent_range = spec["parent_range"]
        parent_shape = spec["parent_shape"]
        weights_subrange = spec["weights_subrange"]
        child_range = spec["child_range"]
        transpose_perm = spec["transpose_perm"]

        if direction == "forward":
            reshape_size = spec["reshape_size"]

            sliced_weights = weights_in[parent_range].reshape(parent_shape)[
                weights_subrange
            ]

            value_to_assign = sliced_weights.transpose(transpose_perm).reshape(
                reshape_size
            )
            weights_out = weights_out.at[child_range].set(value_to_assign)

        elif direction == "backward":
            transpose_child_shape = spec["transpose_child_shape"]
            child_shape = spec["child_shape"]

            sliced_weights = (
                weights_in[child_range]
                .reshape(transpose_child_shape)
                .transpose(transpose_perm)
            )

            value_to_insert = sliced_weights.flatten().reshape(child_shape)

            slab = weights_out[parent_range]
            slab_reshaped = slab.reshape(parent_shape)
            slab_reshaped = slab_reshaped.at[weights_subrange].set(value_to_insert)
            weights_out = weights_out.at[parent_range].set(
                slab_reshaped.reshape(slab.shape)
            )

    return weights_out


def reorder_numpy_jax_helper(schedule, weights_in, direction, has_batch_dim):
    weights_in_jax = jnp.array(weights_in)
    result = reorder_jax_helper(schedule, weights_in_jax, direction, has_batch_dim)
    return np.array(result)


def reorder_jax(schedule, weights_in, direction, has_batch_dim):
    if isinstance(weights_in, (jnp.ndarray, jax.Array)):
        return reorder_jax_helper(schedule, weights_in, direction, has_batch_dim)
    else:
        return reorder_numpy_jax_helper(schedule, weights_in, direction, has_batch_dim)


def clean_tensors(*tensors):
    tensors_clean = []
    for t in tensors:
        result = t
        if type(t) is ad.Zero or ad.is_undefined_primal(t):
            result = jnp.zeros(t.aval.shape, t.aval.dtype)
        tensors_clean.append(result)
    return tensors_clean
