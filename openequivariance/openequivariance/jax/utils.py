import jax
import jax.numpy as jnp
import numpy as np
import functools
import traceback

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


_indentation = 0
def _trace(msg=None):
    """Print a message at current indentation."""
    if msg is not None:
        print("  " * _indentation + msg)

def _trace_indent(msg=None):
    """Print a message and then indent the rest."""
    global _indentation
    _trace(msg)
    _indentation = 1 + _indentation

def _trace_unindent(msg=None):
    """Unindent then print a message."""
    global _indentation
    _indentation = _indentation - 1
    _trace(msg)

def trace(name):
  """A decorator for functions to trace arguments and results."""

  def trace_func(func):  # pylint: disable=missing-docstring
    def pp(v):
        """Print certain values more succinctly"""
        vtype = str(type(v))
        if "jax._src.xla_bridge._JaxComputationBuilder" in vtype:
            return "<JaxComputationBuilder>"
        elif "jaxlib._jax_.XlaOp" in vtype:
            return "<XlaOp at 0x{:x}>".format(id(v))
        elif ("partial_eval.JaxprTracer" in vtype or
              "batching.BatchTracer" in vtype or
              "ad.JVPTracer" in vtype):
            return "Traced<{}>".format(v.aval)
        elif isinstance(v, tuple):
            return "({})".format(pp_values(v))
        else:
            return str(v)
    def pp_values(args):
        return ", ".join([pp(arg) for arg in args])

    @functools.wraps(func)
    def func_wrapper(*args):
      _trace_indent("call {}({})".format(name, pp_values(args)))
      res = func(*args)
      _trace_unindent("|<- {} = {}".format(name, pp(res)))
      return res

    return func_wrapper

  return trace_func

class expectNotImplementedError(object):
  """Context manager to check for NotImplementedError."""
  def __enter__(self): pass
  def __exit__(self, type, value, tb):
    global _indentation
    _indentation = 0
    if type is NotImplementedError:
      print("\nFound expected exception:")
      traceback.print_exc(limit=3)
      return True
    elif type is None:  # No exception
      assert False, "Expected NotImplementedError"
    else:
      return False