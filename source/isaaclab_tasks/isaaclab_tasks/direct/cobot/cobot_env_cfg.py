# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the COBOT two-Franka communication environment."""

from __future__ import annotations

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab.utils.configclass import configclass

from .macro_executor_cfg import MacroExecutorCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


# Station layout constants (env-local frame). Station geometry is IDENTICAL in each
# robot's base frame (pure translation between stations) so that one policy can play
# either seat with symmetric observations.
STATION_Y = 1.35
"""Distance from the dividing wall (y=0) to each robot base [m].

Sized so the SeattleLab table's back-frame gantry clears the wall on both sides;
everything station-local (tables, plates, panel, cameras, robots) derives from this
single constant.
"""

BOX_POS = (0.50, 0.28)
"""Station-local (x, y) of the red/blue target box center [m]."""

BIN_POS = (0.50, -0.28)
"""Station-local (x, y) of the green filler bin center [m]."""

CUBE_SPAWN_RANGE = {"x": (0.35, 0.62), "y": (-0.14, 0.14)}
"""Station-local spawn range for cubes on the table [m]."""

CUBE_HALF_SIZE = 0.02
"""Cube half-extent [m]."""

# Arm + gripper start pose reused from the validated Franka stack task.
_FRANKA_INIT_JOINT_POS: dict[str, float] = {
    "panda_joint1": 0.0444,
    "panda_joint2": -0.1894,
    "panda_joint3": -0.1107,
    "panda_joint4": -2.5148,
    "panda_joint5": 0.0044,
    "panda_joint6": 2.3775,
    "panda_joint7": 0.6952,
    "panda_finger_joint.*": 0.0400,
}


@configclass
class CobotEnvCfg(DirectMARLEnvCfg):
    """Two-Franka COBOT verification environment.

    Two identical stations separated by an opaque wall. Each station has a Franka on
    the validated stack-task table mounting, a red and a blue cube (receiver task), a
    green cube (sender filler task), a box and a bin receptacle, and a cue panel
    behind the base. A per-episode informed arm can read the current target color
    from its panel; the other arm cannot. Selectors act at macro-tick granularity.
    """

    # macro tick timing: one env step = one executor chunk
    decimation = 100
    episode_length_s = 40.0

    # multi-agent spec (spaces filled in __post_init__ from the knobs below)
    possible_agents = ["arm_a", "arm_b"]
    action_spaces = {}
    observation_spaces = {}
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=4,
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysxCfg(bounce_threshold_velocity=0.2),
    )
    viewer: ViewerCfg = ViewerCfg(eye=(3.2, 3.2, 2.6), lookat=(0.4, 0.0, 0.2))

    # scene: heterogeneous per-env content, no physics replication (matches the stack task)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=6.0, replicate_physics=False)

    # robots (station A at +y, station B at -y; both facing +x with identity rotation
    # so base-frame math and world-frame jacobians coincide)
    robot_a_cfg: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/RobotA",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, STATION_Y, 0.0), joint_pos=_FRANKA_INIT_JOINT_POS),
    )
    robot_b_cfg: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/RobotB",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, -STATION_Y, 0.0), joint_pos=_FRANKA_INIT_JOINT_POS),
    )

    # scripted stand-in executor (the frozen VLA slots in behind the same interface)
    executor: MacroExecutorCfg = MacroExecutorCfg()

    # ---- knob 1: rate of change ----
    flip_prob: float = 0.0
    """Per-tick probability that the target color flips mid-episode. 0 = static target."""

    # ---- knob 2: observational asymmetry ----
    asymmetry_p: float = 0.0
    """Per-tick probability that the receiver directly observes the current target color.

    0 = blind receiver (max asymmetry), 1 = fully sighted receiver (communication worthless).
    """

    # ---- knob 3: communication cost geometry ----
    # The cost is physical: reading the cue requires the READ_CUE macro-action, which
    # spends a tick oriented away from the station. Magnitude is tuned via the executor's
    # ``read_pos`` / ``cue_panel_pos`` geometry and ``strict_read``.
    strict_read: bool = True
    """If True, a READ_CUE tick only updates the sender's knowledge when the hand actually
    reached the read pose by tick end. If False, the read always succeeds."""

    # ---- communication channel ----
    comm_mode: str = "channel"
    """One of: "none", "channel", "intent", "obs_broadcast", "oracle". Baselines differ only here."""

    codebook_size: int = 4
    """Number of non-null message symbols K. The action's message head has K+1 options (0 = no-send)."""

    # ---- roles ----
    informed_arm: int | None = None
    """Which arm sees the cue: 0 (arm_a), 1 (arm_b), or None to randomize per episode."""

    observe_role: bool = True
    """Include an is-informed flag in each agent's observation."""

    observe_ticks_since_read: bool = True
    """Include a normalized ticks-since-last-read scalar in the observation."""

    # ---- reward ----
    correct_place_reward: float = 1.0
    """Team reward for placing the cube matching the CURRENT target color in a box."""

    wrong_place_reward: float = 0.0
    """Team reward for placing the non-target color cube in a box (0 = wasted work, no penalty)."""

    green_place_reward: float = 1.0
    """Team reward for placing a green cube in a bin (the sender throughput comm cost eats into)."""

    # ---- visuals / rendering ----
    visual_cue_enabled: bool = False
    """Paint cube/panel colors into USD (displayColor) and keep panels in sync with the
    target. Needed for camera/pixel phases and the occlusion check; irrelevant for
    state-based training."""

    enable_cameras: bool = False
    """Spawn per-station wrist and overview RGB cameras (requires --enable_cameras)."""

    camera_resolution: tuple[int, int] = (200, 200)
    """Camera (width, height) when cameras are enabled [px]."""

    # ---- respawn / detection thresholds ----
    place_radius: float = 0.08
    """Max horizontal distance from receptacle center for a placement to count [m]."""

    place_max_height: float = 0.09
    """Max cube center height for a placement to count [m]."""

    min_cube_separation: float = 0.09
    """Minimum separation between cubes when (re)spawning [m]."""

    def __post_init__(self):
        # observation layout (see CobotEnv._build_agent_obs; keep in sync)
        # ee(3) + grip(1) + held(4) + cubes(9) + instr(4) + fsm(1) + last_read(4)
        # + glimpse(3) + msg(K+1) + progress(1)
        k = self.codebook_size
        base = 3 + 1 + 4 + 9 + 4 + 1 + 4 + 3 + (k + 1) + 1
        if self.observe_ticks_since_read:
            base += 1
        if self.observe_role:
            base += 1
        self.single_obs_dim = base
        obs_dim = base + (base if self.comm_mode == "obs_broadcast" else 0)

        if self.comm_mode == "intent" and k + 1 < 5:
            raise ValueError("comm_mode='intent' needs codebook_size >= 4 to encode the 4 macro-actions.")
        if self.comm_mode not in ("none", "channel", "intent", "obs_broadcast", "oracle"):
            raise ValueError(f"Unknown comm_mode: {self.comm_mode}")

        self.observation_spaces = {agent: obs_dim for agent in self.possible_agents}
        self.action_spaces = {agent: gym.spaces.MultiDiscrete([4, k + 1]) for agent in self.possible_agents}
        # privileged state for a centralized critic: both obs + target one-hot + informed flag
        self.state_space = 2 * obs_dim + 3
