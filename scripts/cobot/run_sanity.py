# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Heuristic-selector rollout for the COBOT two-Franka environment.

Drives both arms with simple hand-coded selectors to sanity-check the rig:
the informed arm reads the cue then farms green cubes while relaying the color;
the receiver follows the received symbol (or guesses when communication is off).
Comparing ``--comm_mode channel`` against ``--comm_mode none`` should show a clear
team-reward gap with a blind receiver (p=0) and a static target.

Also serves as the visualization entry point (small ``--num_envs``, LIVESTREAM/GUI).

.. code-block:: bash

    # sanity comparison (headless)
    ./isaaclab.sh -p scripts/cobot/run_sanity.py --headless --num_envs 64 --steps 400 --comm_mode channel
    ./isaaclab.sh -p scripts/cobot/run_sanity.py --headless --num_envs 64 --steps 400 --comm_mode none

    # visualization (GUI / livestream)
    ./isaaclab.sh -p scripts/cobot/run_sanity.py --num_envs 4 --steps 2000 --visual_cue
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Heuristic rollout for the COBOT two-Franka env.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments.")
parser.add_argument("--steps", type=int, default=400, help="Number of macro ticks to simulate.")
parser.add_argument("--comm_mode", type=str, default="channel", help="none|channel|intent|obs_broadcast|oracle")
parser.add_argument("--flip_prob", type=float, default=0.0, help="Per-tick target flip probability (knob 1).")
parser.add_argument("--asymmetry_p", type=float, default=0.0, help="Receiver glimpse probability (knob 2).")
parser.add_argument("--read_every", type=int, default=8, help="Heuristic sender re-reads the cue every N ticks.")
parser.add_argument("--visual_cue", action="store_true", help="Paint cue/cube colors into USD (for viewing).")
parser.add_argument(
    "--informed_arm", type=int, default=None, choices=[0, 1], help="Fix the informed arm (0=a, 1=b); omit to randomize per episode."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_tasks.direct.cobot.macro_executor import (  # isort: skip
    INSTR_PUT_BLUE_IN_BOX,
    INSTR_PUT_GREEN_IN_BIN,
    INSTR_PUT_RED_IN_BOX,
    INSTR_READ_CUE,
)

TASK = "Isaac-Cobot-TwoFranka-Direct-v0"


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.comm_mode = args_cli.comm_mode
    env_cfg.flip_prob = args_cli.flip_prob
    env_cfg.asymmetry_p = args_cli.asymmetry_p
    env_cfg.visual_cue_enabled = args_cli.visual_cue
    env_cfg.informed_arm = args_cli.informed_arm
    env_cfg.enable_cameras = args_cli.enable_cameras
    env = gym.make(TASK, cfg=env_cfg)
    raw = env.unwrapped

    obs, _ = env.reset()
    n = raw.num_envs
    device = raw.device
    comm_on = args_cli.comm_mode == "channel"

    # what the receiver believes based on delivered symbols (0=unknown, 1=red, 2=blue)
    delivered = torch.zeros(n, 2, dtype=torch.long, device=device)

    total_reward = torch.zeros(n, device=device)
    reward_log = []
    for step in range(args_cli.steps):
        informed = raw.informed_arm  # [n], 0 or 1
        actions = {}
        for i, agent in enumerate(raw.cfg.possible_agents):
            is_informed = informed == i

            # -- sender heuristic: read when stale, otherwise farm greens; always relay last read
            never_read = raw.last_read[:, i] < 2
            stale = raw.ticks_since_read[:, i] >= args_cli.read_every
            sender_macro = torch.where(
                never_read | (stale & (raw.cfg.flip_prob > 0)),
                torch.full((n,), INSTR_READ_CUE, dtype=torch.long, device=device),
                torch.full((n,), INSTR_PUT_GREEN_IN_BIN, dtype=torch.long, device=device),
            )
            sender_msg = torch.where(never_read, torch.zeros(n, dtype=torch.long, device=device), raw.last_read[:, i] - 1)

            # -- receiver heuristic: follow the delivered symbol; guess red when uninformed
            belief = delivered[:, i]
            receiver_macro = torch.where(
                belief == 2,
                torch.full((n,), INSTR_PUT_BLUE_IN_BOX, dtype=torch.long, device=device),
                torch.full((n,), INSTR_PUT_RED_IN_BOX, dtype=torch.long, device=device),
            )
            receiver_msg = torch.zeros(n, dtype=torch.long, device=device)

            macro = torch.where(is_informed, sender_macro, receiver_macro)
            msg = torch.where(is_informed, sender_msg, receiver_msg)
            actions[agent] = torch.stack([macro, msg], dim=-1)

        obs, rewards, terminated, truncated, extras = env.step(actions)

        # symbol delivery for the next tick (honest: only when the channel exists)
        if comm_on:
            for i in range(2):
                incoming = raw.messages[:, 1 - i]
                delivered[:, i] = torch.where(incoming > 0, incoming, delivered[:, i])
        done = truncated[raw.cfg.possible_agents[0]] | terminated[raw.cfg.possible_agents[0]]
        delivered[done] = 0

        r = rewards[raw.cfg.possible_agents[0]]
        total_reward += r
        if done.any():
            reward_log.extend(total_reward[done].tolist())
            total_reward[done] = 0.0
        if step % 40 == 0 and "log" in extras:
            stats = {k: f"{v:.2f}" for k, v in extras["log"].items()}
            print(f"[step {step:5d}] {stats}")

    if reward_log:
        mean_r = sum(reward_log) / len(reward_log)
        print(f"\n=== comm_mode={args_cli.comm_mode} flip_prob={args_cli.flip_prob} p={args_cli.asymmetry_p} ===")
        print(f"episodes completed: {len(reward_log)}")
        print(f"mean episode team reward: {mean_r:.3f}")
    else:
        print("No full episodes completed - increase --steps.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
