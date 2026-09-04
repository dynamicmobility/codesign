import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict
from flax import traverse_util
from typing import Literal, Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
import numpy as np
from typing import Sequence

from codesign.utils.model import normalize_design, sample_designs

class MLP(nn.Module):
    obs_dim: int                
    hidden_sizes: Sequence[int] 
    action_dim: int             

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, self.obs_dim)

        # Hidden layers
        for h in self.hidden_sizes:
            # x = nn.LayerNorm()(x)
            x = nn.Dense(h)(x)
            x = nn.swish(x)

        # Output layer
        x = nn.Dense(self.action_dim)(x)
        return x

def flatten_model(target_params):
    flat_params_list = []
    num_params = 0
    target_policy_info = []

    def flatten_dict(d, prefix=""):
        items = []
        for k, v in d.items():
            if isinstance(v, (dict, FrozenDict)):
                items.extend(flatten_dict(v, prefix + k + "/"))
            else:
                items.append((prefix + k, v))
        items.sort()
        return items

    for name, param in flatten_dict(target_params):
        shape = param.shape
        target_policy_info.append((name, shape))
        num_params += np.prod(shape)
        flat_params_list.append(param.reshape(-1))

    flat_init = jnp.concatenate(flat_params_list)

    return flat_init, num_params, target_policy_info

class DualA2CHypernet(nn.Module):
    target_policy_dict: dict
    target_value_dict: dict
    num_objs: int
    obs_dim: int
    hypersize: tuple = (128, 128)
    num_features: int = 8
    W_variance: float = 0.0

    def setup(self):
        # Init target policy params
        target_policy_params = self.target_policy_dict['params']
        target_value_params  = self.target_value_dict['params']

        target_policy_info = []
        target_value_info  = []

        policy_flat_init, num_policy_params, target_policy_info = flatten_model(target_policy_params)
        value_flat_init, num_value_params, target_value_info    = flatten_model(target_value_params)
        
        # Save as immutable attributes
        self.target_policy_info = tuple(target_policy_info)
        self.target_value_info  = tuple(target_value_info)
        self.num_policy_params  = num_policy_params
        self.num_value_params   = num_value_params

        # Hypernetwork MLP
        self.policy_mlp = MLP(
            obs_dim      = self.num_objs,
            hidden_sizes = self.hypersize,
            action_dim   = self.num_features
        )
        self.value_mlp = MLP(
            obs_dim      = self.num_objs,
            hidden_sizes = self.hypersize,
            action_dim   = self.num_features
        )

        # Hypernet params
        self.policy_W = self.param(
            "policy_W",
            lambda key: (
                jax.random.uniform(key, (self.num_features, self.num_policy_params), minval=-1, maxval=1)
                * self.W_variance
            ),
        )
        self.policy_b = self.param(
            "policy_b",
            lambda key: policy_flat_init
        )
        
        # Hypernet params
        self.value_W = self.param(
            "value_W",
            lambda key: (
                jax.random.uniform(key, (self.num_features, self.num_value_params), minval=-1, maxval=1)
                * self.W_variance
            ),
        )
        self.value_b = self.param(
            "value_b",
            lambda key: value_flat_init
        )

    def unflatten_params(self, flat_params, target_network, single):
        # Reshape back into dict
        sorted_params = {}
        cnt = 0
        for name, shape in target_network:
            size = np.prod(shape)
            sorted_params[name] = flat_params[..., cnt:cnt+size].reshape((-1,) + shape)
            cnt += size

        nested_params = traverse_util.unflatten_dict({
            tuple(k.split("/")): v if not single else v[0] for k, v in sorted_params.items()
        })
        nested_params = FrozenDict(nested_params)

        # Wrap under "params" collection for Flax
        return {"params": nested_params}

    def __call__(self, pref):
        policy_features = self.policy_mlp(pref)
        value_features  = self.value_mlp(pref)
        policy_flat_params = jnp.dot(policy_features, self.policy_W) + self.policy_b
        value_flat_params  = jnp.dot(value_features,  self.value_W) + self.value_b

        policy_sort_params = self.unflatten_params(
            flat_params    = policy_flat_params,
            target_network = self.target_policy_info,
            single         = len(pref.shape) == 1
        )

        value_sort_params  = self.unflatten_params(
            flat_params    = value_flat_params,
            target_network = self.target_value_info,
            single         = len(pref.shape) == 1
        )

        return ((policy_sort_params, value_sort_params),
                (policy_flat_params, value_flat_params),
                (policy_features, value_features))

def sobol_design_table(seed: int, num_designs: int, low, high) -> np.ndarray:
    """The ``M`` fixed designs a :class:`LookupA2CHypernet` looks up, ``(M, design_dim)``.
    """
    low, high = np.asarray(low, np.float32), np.asarray(high, np.float32)
    physical = sample_designs(
        np.random.default_rng(seed), num_designs, low=low, high=high, dim=len(low)
    )
    return np.asarray(normalize_design(jnp.asarray(physical), low, high))


class LookupA2CHypernet(DualA2CHypernet):
    """A hypernetwork whose feature map is a one-hot lookup over ``design_table``.

    With ``features(d_i) = e_i`` the forward pass collapses to ``flat(d_i) = W[i] + b``, so
    row ``i`` of each ``W`` holds design ``i``'s own policy and

        dL/dW[j] = sum_n features_j(d_n) * dL/dflat_n

    is fed by design ``j``'s rollouts alone. ``num_features`` must equal ``len(design_table)``.

    ``argmin`` is integer-valued, so no gradient reaches ``design``; an algo that
    differentiates through the design cannot use this module.

    Attributes:
        design_table: ``M`` normalized designs, as nested tuples.
        freeze_bias: hold ``b`` fixed. ``dL/db`` sums over every design, so a trainable ``b``
            moves all ``M`` policies on every step.
    """

    design_table: tuple = ()
    freeze_bias: bool = True

    def setup(self):
        super().setup()
        self.table = jnp.asarray(self.design_table, jnp.float32)  # (M, design_dim)

    def features(self, design):
        """One-hot row of the nearest table entry, ``(batch, M)``."""
        design = design.reshape(-1, self.table.shape[-1])
        distance = jnp.sum((design[:, None] - self.table[None]) ** 2, axis=-1)
        return jax.nn.one_hot(jnp.argmin(distance, axis=-1), self.table.shape[0])

    def __call__(self, design):
        features = self.features(design)
        hold = jax.lax.stop_gradient if self.freeze_bias else (lambda x: x)
        policy_flat_params = jnp.dot(features, self.policy_W) + hold(self.policy_b)
        value_flat_params = jnp.dot(features, self.value_W) + hold(self.value_b)

        single = len(design.shape) == 1
        return (
            (self.unflatten_params(policy_flat_params, self.target_policy_info, single),
             self.unflatten_params(value_flat_params, self.target_value_info, single)),
            (policy_flat_params, value_flat_params),
            (features, features),
        )
