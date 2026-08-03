import json
import types as _types
import typing
from pathlib import Path
import minimal_mjx as mm

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, model_validator


def _annotation_types(annotation):
    """Flatten a type annotation into its constituent types (unpacking unions)."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is getattr(_types, 'UnionType', ()):
        out = []
        for arg in typing.get_args(annotation):
            out.extend(_annotation_types(arg))
        return out
    return [annotation]


def _subclasses(cls):
    """A Config type and every type derived from it, recursively."""
    out = [cls]
    for sub in cls.__subclasses__():
        out.extend(_subclasses(sub))
    return out


class Config(BaseModel):
    # numpy arrays aren't pydantic-native types; allow them through as-is.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='after')
    def validate(self):
        # Subclasses override with their own invariant checks.
        return self

    @classmethod
    def assert_equals_or_none(cls, atr, ref):
        """Asserts atr == ref OR ref is None"""
        assert (atr == ref) or (ref is None)

    def assert_positive(self, *names):
        """Asserts the named fields are > 0, elementwise for array/sequence fields.
        With no names, checks every numeric field (bools excluded)."""
        def numeric(value):
            return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)

        if not names:
            names = [n for n in type(self).model_fields
                     if numeric(getattr(self, n))]
        for name in names:
            value = getattr(self, name)
            assert np.all(np.asarray(value) > 0), f'{type(self).__name__}.{name} must be > 0, got {value}'

    # ------------------------------------------------------------------ #
    # JSON (de)serialization                                             #
    # ------------------------------------------------------------------ #
    def to_jsonable_dict(self):
        """Return a JSON-safe dict (nested Configs and numpy arrays flattened)."""
        def convert(obj):
            if isinstance(obj, Config):
                return {name: convert(getattr(obj, name)) for name in type(obj).model_fields}
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            # Modules, classes and callables have no JSON form; keep their bare
            # name (a module's __name__ is its full dotted path).
            return getattr(obj, '__name__', str(obj)).rsplit('.', 1)[-1]
        return convert(self)

    def to_json_string(self, **kwargs):
        """Serialize this config to a JSON string. Extra kwargs go to json.dumps."""
        return json.dumps(self.to_jsonable_dict(), **kwargs)

    def save_json_path(self, path):
        """Write this config to `path` as JSON. Returns the Path written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json_string(indent=2))
        return path

    @classmethod
    def from_jsonable_dict(cls, data):
        """Reconstruct a config from a plain dict (inverse of `to_jsonable_dict`)."""
        kwargs = {}
        for name, field in cls.model_fields.items():
            if name not in data:
                continue
            kwargs[name] = cls._decode_field(data[name], field.annotation)
        return cls(**kwargs)

    @classmethod
    def from_json_string(cls, s):
        """Reconstruct a config from a JSON string."""
        return cls.from_jsonable_dict(json.loads(s))

    @classmethod
    def load_json_path(cls, path):
        """Load a config from a JSON file at `path`."""
        return cls.from_jsonable_dict(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------------ #
    # YAML (de)serialization                                             #
    # ------------------------------------------------------------------ #
    def to_yaml_string(self, **kwargs):
        """Serialize this config to a YAML string. Extra kwargs go to yaml.safe_dump."""
        kwargs.setdefault('sort_keys', False)
        kwargs.setdefault('default_flow_style', False)
        return yaml.safe_dump(self.to_jsonable_dict(), **kwargs)

    def save_yaml_path(self, path):
        """Write this config to `path` as YAML. Returns the Path written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml_string())
        return path

    @classmethod
    def load_yaml_path(cls, path):
        """Load a config from a YAML file at `path`."""
        return cls.from_jsonable_dict(yaml.safe_load(Path(path).read_text()))

    @staticmethod
    def _decode_field(value, annotation):
        """Change a JSON value back to the type expected by a field annotation."""
        candidates = _annotation_types(annotation)
        # Nested Config models. 
        if isinstance(value, dict):
            # Subclasses too: a field typed `Robot` may hold a `Cheetah`.
            configs = [s for t in candidates
                       if isinstance(t, type) and issubclass(t, Config)
                       for s in _subclasses(t)]
            if configs:
                keys = set(value)
                best = max(configs, key=lambda t: (
                    set(t.model_fields) == keys, len(keys & set(t.model_fields))
                ))
                return best.from_jsonable_dict(value)
        # Lists: keep as list if a list type is expected, else rebuild ndarray.
        if isinstance(value, list):
            expects_list = any(
                t is list or typing.get_origin(t) is list for t in candidates
            )
            if not expects_list and any(t is np.ndarray for t in candidates):
                return np.array(value)
        return value
    
class Objectives(Config):
    objectives              : list[list[str]]
    shared_objectives       : list[str]
    labels                  : list[str]
    default_scalarization   : list[float]
    
class Weights(Config):
    pass

class Sigmas(Config):
    pass

    def validate(self):
        self.assert_positive()
        return super().validate()
    
class Reward(Config):
    weights       : Weights
    sigmas        : Sigmas | None = None
    objectives    : Objectives

class Robot(Config):
    ctrl_dt         : float
    sim_dt          : float
    action_scale    : float
    termination     : bool | typing.Any = None
    reward          : Reward
    objectives      : Objectives | None = None

    def validate(self):
        self.assert_positive()
        return super().validate()
    

class PPO(Config):
    num_timesteps           : int
    episode_length          : int
    num_envs                : int
    unroll_length           : int
    batch_size              : int
    num_minibatches         : int
    num_updates_per_batch   : int
    learning_rate           : float
    entropy_cost            : float
    discounting             : float
    reward_scaling          : float
    clipping_epsilon        : float
    gae_lambda              : float
    max_grad_norm           : float
    normalize_advantage     : bool
    normalize_observations  : bool
    num_evals               : int
    num_eval_envs           : int
    deterministic_eval      : bool
    seed                    : int
    
    def validate(self):
        # entropy_cost, discounting, gae_lambda and seed are all valid at 0.
        assert(self.num_timesteps > 0)
        assert(self.episode_length > 0)
        assert(self.num_envs > 0)
        assert(self.unroll_length > 0)
        assert(self.batch_size > 0)
        assert(self.num_minibatches > 0)
        assert(self.num_updates_per_batch > 0)
        assert(self.learning_rate > 0)
        assert(self.reward_scaling > 0)
        assert(self.max_grad_norm > 0)
        assert(self.num_evals > 0)
        assert(self.num_eval_envs > 0)
        # Discount and GAE trace decay are both convex weightings over time.
        assert(0 <= self.discounting <= 1)
        assert(0 <= self.gae_lambda <= 1)
        # Above 1 the ratio clip stops binding for any plausible update.
        assert(0 < self.clipping_epsilon <= 1)
        # One training step gathers batch_size * num_minibatches env-steps' worth
        # of data in num_envs-wide scans, so the split must be whole.
        assert (self.batch_size * self.num_minibatches) % self.num_envs == 0, (
            'batch_size * num_minibatches must be divisible by num_envs'
        )
        return super().validate()
    
class ActorCriticNetworks(Config):
    policy_hidden_layer_sizes: list[int]
    value_hidden_layer_sizes:  list[int]
    
class ACHypernetwork(ActorCriticNetworks):
    encoder_size: list[int]
    num_features: int
    
class Design(Config):
    name      : str
    setter    : typing.Callable
    low       : float
    high      : float

class Codesign(Config):
    default_design    : list[float]
    designs           : list[Design]

class MultiObjective(Config):
    num_tradeoffs   : int
    alpha           : float
    sampling        : str

class BasePPO(Config):
    ppo_params        : PPO
    network_params    : ActorCriticNetworks
    
class CodesignHypernetwork(BasePPO):
    codesign: Codesign
    
class MOCOdesignHypernetwork(BasePPO):
    mo          : MultiObjective
    codesign    : Codesign
    
class Train(Config):
    name: str
    save_dir: str
    description: str | None = None
    algorithm: str
    env: str
    backend: str
    env_config: Robot
    learning: BasePPO
    
    @classmethod
    def from_legacy_config(cls, config):
        config = mm.read_config(config)
        
