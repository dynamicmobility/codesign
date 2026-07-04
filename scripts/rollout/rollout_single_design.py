"""Record a trained design-hypernetwork policy on a single design (single-threaded).

Dispatches on the run's ``config["algorithm"]`` so one script handles both network types:

  * ``design_hypernetwork``    -> ``H(d)``    : conditioned on the design only.
  * ``mo_design_hypernetwork`` -> ``H(d, w)`` : conditioned on the design *and* a tradeoff
    (objective scalarization) ``w``.

The script always accepts both a ``--design`` and an objective-scalarization ``--tradeoff``
argument (each with a default). For the single-objective ``design_hypernetwork``, ``H(d)``
does not consume a tradeoff, so ``--tradeoff`` is ignored there (a note is printed).
"""
import os
os.environ["MUJOCO_GL"] = "egl"
# Force JAX to use CPU (must be set before importing jax)
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import argparse
from pathlib import Path

import minimal_mjx as mm
import matplotlib.pyplot as plt
import moplayground as mop
from codesign.envs.MAICheetah import MAICheetah
from codesign.eval import (
    rollout_design_hypernetwork_video,
    rollout_mo_design_hypernetwork_video,
)

CONFIG_PATH = "config/design_hypernetwork_cheetah.yaml"

D = 1.0          # back-leg length scale to render
W = None         # objective scalarization (defaults to uniform over the env's objectives)
T = 500          # rollout length (env steps)
OUT_DIR = Path("scripts/outputs")


def main(config_path: str, checkpoint_path: str | None, design: float,
         tradeoff: list[float] | None, steps: int, camera: str) -> None:
    config = mm.utils.config.create_config_dict(mop.utils.read_config(config_path))
    env_params = mm.utils.config.create_config_dict(config["env_config"])
    env = MAICheetah(env_params=env_params, backend="np")

    algorithm = config["algorithm"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if algorithm == "mo_design_hypernetwork":
        opt = config["env_config"]["reward"]["optimization"]
        objectives = list(opt["objectives"])
        labels = list(opt.get("labels", objectives))
        num_obj = len(objectives)
        if tradeoff is None:
            tradeoff = [1.0 / num_obj] * num_obj
        if len(tradeoff) != num_obj:
            raise ValueError(
                f"--tradeoff must have {num_obj} entries (one per objective); "
                f"got {len(tradeoff)}: {tradeoff}"
            )
        norm = [w / sum(tradeoff) for w in tradeoff]
        print("objective scalarization (raw -> normalized):")
        for label, obj, raw, w in zip(labels, objectives, tradeoff, norm):
            print(f"  {label} {obj}: {raw:g} -> {w:.3f}")
        frames, traj, reward_plotter, _, _ = rollout_mo_design_hypernetwork_video(
            env, config, design=design, tradeoff=tradeoff, n_steps=steps,
            checkpoint_path=checkpoint_path, camera=camera, width=640, height=480,
        )
        out = OUT_DIR / f"mo_design_hypernetwork.mp4"
        title = f"MAI Cheetah d={design} w={tradeoff} reward"
    elif algorithm == "design_hypernetwork":
        if tradeoff is not None:
            print("note: H(d) is single-objective; --tradeoff is ignored.")
        frames, traj, reward_plotter, _, _ = rollout_design_hypernetwork_video(
            env, config, design=design, n_steps=steps,
            checkpoint_path=checkpoint_path, camera=camera, width=640, height=480,
        )
        out = OUT_DIR / f"design_hypernetwork.mp4"
        title = f"MAI Cheetah d={design} reward"
    else:
        raise ValueError(
            f"unsupported algorithm {algorithm!r}; expected 'design_hypernetwork' or "
            "'mo_design_hypernetwork'."
        )

    mm.utils.plotting.save_video(frames, env.dt, out)
    print(f"rendered {len(traj)} steps for design d={design} -> {out}")
    
    reward_plotter.plot(title=title)
    plt.savefig(out.with_suffix(".pdf"))
    print(f"rendered plots -> {out.with_suffix(".pdf")}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="explicit checkpoint dir; defaults to latest under save_dir/name",
    )
    parser.add_argument("--design", type=float, default=D, help="back-leg length scale")
    parser.add_argument(
        "--tradeoff", type=float, nargs="+", default=W,
        help="objective scalarization w (one weight per objective; normalized onto the "
             "simplex). Only used for mo_design_hypernetwork; defaults to uniform.",
    )
    parser.add_argument("--steps", type=int, default=T, help="rollout length (env steps)")
    parser.add_argument("--camera", type=str, default="track", help="render camera name")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.design, args.tradeoff, args.steps, args.camera)
