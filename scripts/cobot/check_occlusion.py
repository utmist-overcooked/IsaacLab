# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Occlusion verification for the COBOT information gap (invariant 3).

Paints EVERYTHING at the informed station (cue panel + all three cubes) a sentinel
magenta that appears nowhere else in the scene, then rolls the receiver arm through
random macro-actions while capturing its wrist and station cameras. Any magenta pixel
in a receiver frame is a leak through/around the dividing wall.

Must run with cameras: pass ``--enable_cameras``.

.. code-block:: bash

    ./isaaclab.sh -p scripts/cobot/check_occlusion.py --headless --enable_cameras \
        --num_envs 4 --steps 60 --outdir /scratch/$USER/cobot_occlusion
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="COBOT occlusion check.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
parser.add_argument("--steps", type=int, default=60, help="Macro ticks to roll and capture.")
parser.add_argument("--outdir", type=str, default=None, help="Where to save sample frames (optional).")
parser.add_argument("--tolerance", type=float, default=0.25, help="Per-channel color match tolerance (0-1).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_tasks.direct.cobot.cobot_env as cobot_env_module  # isort: skip

TASK = "Isaac-Cobot-TwoFranka-Direct-v0"
SENTINEL = (1.0, 0.0, 1.0)  # magenta: appears nowhere else in the scene


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.visual_cue_enabled = True
    env_cfg.enable_cameras = True
    env_cfg.informed_arm = 0  # station A is informed; arm_b's cameras must never see it
    env_cfg.comm_mode = "none"
    env = gym.make(TASK, cfg=env_cfg)
    raw = env.unwrapped

    env.reset()

    # paint the informed station's panel and cubes with the sentinel color
    from pxr import Usd, UsdGeom

    stage = raw.sim.stage

    def paint(path: str):
        prim = stage.GetPrimAtPath(path)
        for p in Usd.PrimRange(prim):
            if p.IsA(UsdGeom.Gprim):
                UsdGeom.Gprim(p).GetDisplayColorAttr().Set([SENTINEL])
                return
        raise RuntimeError(f"no gprim under {path}")

    # with informed_arm=0, station A is the sender's: its panel, green cube, and bin
    # are the far-station content the receiver (arm_b) must never see
    for i in range(raw.num_envs):
        paint(f"/World/envs/env_{i}/CuePanel")
        paint(f"/World/envs/env_{i}/CubeGreen")
        paint(f"/World/envs/env_{i}/BinPlate")
    # keep the env from repainting the panel over the sentinel
    raw.cfg.visual_cue_enabled = False

    sentinel = torch.tensor(SENTINEL)
    tol = args_cli.tolerance
    max_hits = 0
    worst = None

    if args_cli.outdir:
        os.makedirs(args_cli.outdir, exist_ok=True)

    for step in range(args_cli.steps):
        actions = {
            agent: torch.stack(
                [
                    torch.randint(0, 4, (raw.num_envs,), device=raw.device),
                    torch.zeros(raw.num_envs, dtype=torch.long, device=raw.device),
                ],
                dim=-1,
            )
            for agent in raw.cfg.possible_agents
        }
        env.step(actions)

        for cam_name in ("wrist_cam_b", "station_cam_b"):
            rgb = raw.scene.sensors[cam_name].data.output["rgb"]  # [n, h, w, c]
            rgb = rgb[..., :3].float().cpu()
            if rgb.max() > 1.5:
                rgb = rgb / 255.0
            match = (rgb - sentinel).abs().max(dim=-1).values < tol  # [n, h, w]
            hits = int(match.sum().item())
            if hits > max_hits:
                max_hits = hits
                worst = (step, cam_name, int(match.sum(dim=(1, 2)).argmax().item()))
            if args_cli.outdir and (hits > 0 or step % 20 == 0):
                from PIL import Image

                frame = (rgb[0].clamp(0, 1) * 255).to(torch.uint8).numpy()
                Image.fromarray(frame).save(f"{args_cli.outdir}/{cam_name}_step{step:04d}.png")

    print("\n=== occlusion check (receiver = arm_b, informed station A painted magenta) ===")
    if max_hits == 0:
        print(f"PASS: no sentinel pixels in any receiver frame over {args_cli.steps} ticks x {raw.num_envs} envs.")
    else:
        print(f"FAIL: up to {max_hits} sentinel pixels leaked (first worst at step/cam/env = {worst}).")
        print("The wall or camera placement lets the receiver see the cue or far-station cubes.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
