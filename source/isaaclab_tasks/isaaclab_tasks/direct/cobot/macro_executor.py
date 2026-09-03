# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted macro-action executor for the COBOT two-Franka task.

This module implements the stand-in low-level executor described in the COBOT
environment plan: a vectorized finite-state machine that drives one Franka arm
through a full pick-and-place cycle (or a cue-read reorientation) given a
discrete macro-action instruction. It fills the interface slot where a frozen
VLA policy will later sit: instructions in, joint-position targets out, with
the environment tick as the replanning boundary.

The FSM is intentionally plain torch (no warp) and consumes/produces only
robot-local quantities, so it has no dependency on the scene layout beyond the
waypoints passed in through :class:`MacroExecutorCfg`.
"""

from __future__ import annotations

import math
import torch

from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

from .macro_executor_cfg import MacroExecutorCfg

# Macro-action instruction ids (must match the env's action space ordering).
INSTR_PUT_RED_IN_BOX = 0
INSTR_PUT_BLUE_IN_BOX = 1
INSTR_PUT_GREEN_IN_BIN = 2
INSTR_READ_CUE = 3
INSTR_NONE = -1

# Cube indices within a station (fixed ordering used across env and executor).
CUBE_RED = 0
CUBE_BLUE = 1
CUBE_GREEN = 2


class FrankaMacroExecutor:
    """Vectorized scripted executor for one Franka arm across all envs.

    One instance drives one of the two station arms. Each physics step,
    :meth:`step` advances the FSM and writes joint-position targets to the
    articulation via a differential IK controller. Instructions change only at
    macro-tick boundaries through :meth:`set_instructions`.

    FSM states::

        IDLE -> GO_ABOVE -> DESCEND -> GRASP -> LIFT -> GO_RECEP -> LOWER -> RELEASE -> RETREAT -> IDLE
        READ (terminal within a tick; entered/held while the instruction is READ_CUE)
    """

    IDLE = 0
    GO_ABOVE = 1
    DESCEND = 2
    GRASP = 3
    LIFT = 4
    GO_RECEP = 5
    LOWER = 6
    RELEASE = 7
    RETREAT = 8
    READ = 9

    def __init__(
        self,
        cfg: MacroExecutorCfg,
        robot: Articulation,
        base_pos_w: torch.Tensor,
        box_pos: tuple[float, float],
        bin_pos: tuple[float, float],
        physics_dt: float,
        num_envs: int,
        device: str,
    ):
        """Initialize the executor.

        Args:
            cfg: Waypoint and timing parameters.
            robot: The arm articulation to drive.
            base_pos_w: World position of the robot base per env, shape [num_envs, 3], [m].
            box_pos: Station-local (x, y) of the red/blue box center [m].
            bin_pos: Station-local (x, y) of the green bin center [m].
            physics_dt: Physics step size [s].
            num_envs: Number of parallel envs.
            device: Torch device.
        """
        self.cfg = cfg
        self.robot = robot
        self.num_envs = num_envs
        self.device = device
        self._dt = physics_dt
        self._base_pos_w = base_pos_w

        # joint indexing
        self.arm_joint_ids = [robot.joint_names.index(f"panda_joint{i}") for i in range(1, 8)]
        self.finger_joint_ids = [
            robot.joint_names.index(name) for name in ("panda_finger_joint1", "panda_finger_joint2")
        ]
        self.hand_body_id = robot.body_names.index("panda_hand")
        # for a fixed-base robot the root body is excluded from the jacobian rows
        self.ee_jacobi_idx = self.hand_body_id - 1 if robot.is_fixed_base else self.hand_body_id
        self.jacobi_joint_ids = [j + robot.num_base_dofs for j in self.arm_joint_ids]

        # IK controller
        ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.ik = DifferentialIKController(ik_cfg, num_envs=num_envs, device=device)

        # fixed orientations (xyzw)
        self.down_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(num_envs, 1)
        read_dir = torch.tensor(cfg.cue_panel_pos[:2], device=device) - torch.tensor(cfg.read_pos[:2], device=device)
        azimuth = torch.atan2(read_dir[1], read_dir[0]).repeat(num_envs)
        zeros = torch.zeros(num_envs, device=device)
        pitch = torch.full((num_envs,), math.pi / 2.0, device=device)
        # hand z-axis horizontal, pointing at the panel: Rz(azimuth) * Ry(pi/2)
        self.read_quat = quat_mul(quat_from_euler_xyz(zeros, zeros, azimuth), quat_from_euler_xyz(zeros, pitch, zeros))

        # receptacle waypoints (station-local xy)
        self._box_pos = torch.tensor(box_pos, device=device).repeat(num_envs, 1)
        self._bin_pos = torch.tensor(bin_pos, device=device).repeat(num_envs, 1)
        self._home_pos = torch.tensor(cfg.home_pos, device=device).repeat(num_envs, 1)
        self._read_pos = torch.tensor(cfg.read_pos, device=device).repeat(num_envs, 1)

        # timing in steps
        self._timeout_steps = int(cfg.state_timeout_s / physics_dt)
        self._grasp_steps = int(cfg.grasp_dwell_s / physics_dt)
        self._release_steps = int(cfg.release_dwell_s / physics_dt)

        # FSM buffers
        self.state = torch.full((num_envs,), self.IDLE, dtype=torch.long, device=device)
        self.timer = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.instruction = torch.full((num_envs,), INSTR_NONE, dtype=torch.long, device=device)

    """
    Tick-boundary and reset operations.
    """

    def set_instructions(self, instructions: torch.Tensor):
        """Apply new instructions at a macro-tick boundary.

        A changed instruction interrupts the running FSM and restarts it for the new
        subtask. An unchanged instruction continues the running FSM; if the FSM had
        finished (IDLE), it restarts on a fresh cube.

        Args:
            instructions: Instruction ids, shape [num_envs], long.
        """
        changed = instructions != self.instruction
        restart = changed | (self.state == self.IDLE)
        is_read = instructions == INSTR_READ_CUE
        new_state = torch.where(is_read, torch.full_like(self.state, self.READ), torch.full_like(self.state, self.GO_ABOVE))
        self.state = torch.where(restart, new_state, self.state)
        self.timer = torch.where(restart, torch.zeros_like(self.timer), self.timer)
        self.instruction = instructions.clone()

    def reset_idx(self, env_ids: torch.Tensor):
        """Reset the FSM for the given envs."""
        self.state[env_ids] = self.IDLE
        self.timer[env_ids] = 0
        self.instruction[env_ids] = INSTR_NONE
        self.ik.reset(env_ids)

    """
    Per-physics-step operation.
    """

    def step(self, cube_pos_local: torch.Tensor):
        """Advance the FSM one physics step and write joint targets to the robot.

        Args:
            cube_pos_local: Positions of this station's cubes in the robot base frame,
                shape [num_envs, 3, 3] indexed [env, cube(red/blue/green), xyz], [m].
        """
        cfg = self.cfg
        n = self.num_envs

        # current hand pose in base frame (base has identity orientation by construction)
        hand_pose_w = self.robot.data.body_pose_w.torch[:, self.hand_body_id]
        hand_pos = hand_pose_w[:, 0:3] - self._base_pos_w
        hand_quat = hand_pose_w[:, 3:7]
        tcp_pos = hand_pos + quat_apply(hand_quat, torch.tensor([0.0, 0.0, cfg.tcp_offset], device=self.device).repeat(n, 1))

        # target cube / receptacle by instruction
        cube_idx = self.instruction.clamp(min=CUBE_RED, max=CUBE_GREEN)
        target_cube = cube_pos_local[torch.arange(n, device=self.device), cube_idx]
        recep_xy = torch.where((self.instruction == INSTR_PUT_GREEN_IN_BIN).unsqueeze(-1), self._bin_pos, self._box_pos)

        # per-state desired hand pose and gripper command
        grasp_z = target_cube[:, 2] + cfg.grasp_height_offset + cfg.tcp_offset
        des_pos = self._home_pos.clone()
        des_quat = self.down_quat.clone()
        gripper_open = torch.ones(n, dtype=torch.bool, device=self.device)

        s = self.state
        above = torch.cat([target_cube[:, 0:2], torch.full((n, 1), cfg.approach_height, device=self.device)], dim=-1)
        at_cube = torch.cat([target_cube[:, 0:2], grasp_z.unsqueeze(-1)], dim=-1)
        carry = torch.cat([tcp_pos[:, 0:2], torch.full((n, 1), cfg.carry_height, device=self.device)], dim=-1)
        over_recep = torch.cat([recep_xy, torch.full((n, 1), cfg.carry_height, device=self.device)], dim=-1)
        at_recep = torch.cat([recep_xy, torch.full((n, 1), cfg.drop_height, device=self.device)], dim=-1)

        des_pos = torch.where((s == self.GO_ABOVE).unsqueeze(-1), above, des_pos)
        des_pos = torch.where((s == self.DESCEND).unsqueeze(-1), at_cube, des_pos)
        des_pos = torch.where((s == self.GRASP).unsqueeze(-1), at_cube, des_pos)
        des_pos = torch.where((s == self.LIFT).unsqueeze(-1), carry, des_pos)
        des_pos = torch.where((s == self.GO_RECEP).unsqueeze(-1), over_recep, des_pos)
        des_pos = torch.where((s == self.LOWER).unsqueeze(-1), at_recep, des_pos)
        des_pos = torch.where((s == self.RELEASE).unsqueeze(-1), at_recep, des_pos)
        des_pos = torch.where((s == self.RETREAT).unsqueeze(-1), over_recep, des_pos)
        des_pos = torch.where((s == self.READ).unsqueeze(-1), self._read_pos, des_pos)
        des_quat = torch.where((s == self.READ).unsqueeze(-1), self.read_quat, des_quat)

        closed_states = (s == self.GRASP) | (s == self.LIFT) | (s == self.GO_RECEP) | (s == self.LOWER)
        gripper_open = ~closed_states

        # a pick instruction whose target cube is outside the workspace (partner's table)
        # is a no-op: hold the home pose instead of chasing it
        pick_instr = (self.instruction >= INSTR_PUT_RED_IN_BOX) & (self.instruction <= INSTR_PUT_GREEN_IN_BIN)
        unreachable = pick_instr & (torch.norm(target_cube[:, 0:2], dim=-1) > 0.85)
        des_pos = torch.where(unreachable.unsqueeze(-1), self._home_pos, des_pos)
        des_quat = torch.where(unreachable.unsqueeze(-1), self.down_quat, des_quat)
        gripper_open = gripper_open | unreachable

        # command targets: hand pose target is the FSM waypoint (positions are hand-frame heights)
        # note: waypoints above are TCP-centric in xy and hand-centric in z where grasp_z already
        # includes the tcp offset; approach/carry heights are hand heights by definition.
        self.ik.set_command(torch.cat([des_pos, des_quat], dim=-1))
        jacobian = self.robot.data.body_link_jacobian_w.torch[:, self.ee_jacobi_idx, :, :][:, :, self.jacobi_joint_ids]
        joint_pos = self.robot.data.joint_pos.torch[:, self.arm_joint_ids]
        joint_targets = self.ik.compute(hand_pos, hand_quat, jacobian, joint_pos)
        self.robot.set_joint_position_target_index(target=joint_targets, joint_ids=self.arm_joint_ids)

        finger_target = torch.where(gripper_open, cfg.gripper_open, cfg.gripper_close)
        self.robot.set_joint_position_target_index(
            target=finger_target.unsqueeze(-1).repeat(1, 2), joint_ids=self.finger_joint_ids
        )

        # transitions
        self.timer += 1
        pos_err = torch.norm(des_pos - hand_pos, dim=-1)
        reached = pos_err < cfg.pos_threshold
        timed_out = self.timer >= self._timeout_steps

        adv = torch.zeros(n, dtype=torch.bool, device=self.device)
        motion_states = (s == self.GO_ABOVE) | (s == self.DESCEND) | (s == self.LIFT) | (s == self.GO_RECEP) | (
            s == self.LOWER
        ) | (s == self.RETREAT)
        adv = adv | (motion_states & (reached | timed_out))
        adv = adv | ((s == self.GRASP) & (self.timer >= self._grasp_steps))
        adv = adv | ((s == self.RELEASE) & (self.timer >= self._release_steps))

        next_state = s.clone()
        next_state = torch.where(adv & (s == self.GO_ABOVE), torch.full_like(s, self.DESCEND), next_state)
        next_state = torch.where(adv & (s == self.DESCEND), torch.full_like(s, self.GRASP), next_state)
        next_state = torch.where(adv & (s == self.GRASP), torch.full_like(s, self.LIFT), next_state)
        next_state = torch.where(adv & (s == self.LIFT), torch.full_like(s, self.GO_RECEP), next_state)
        next_state = torch.where(adv & (s == self.GO_RECEP), torch.full_like(s, self.LOWER), next_state)
        next_state = torch.where(adv & (s == self.LOWER), torch.full_like(s, self.RELEASE), next_state)
        next_state = torch.where(adv & (s == self.RELEASE), torch.full_like(s, self.RETREAT), next_state)
        next_state = torch.where(adv & (s == self.RETREAT), torch.full_like(s, self.IDLE), next_state)

        # dropped the cube mid-carry: restart the pick
        carrying = (s == self.LIFT) | (s == self.GO_RECEP) | (s == self.LOWER)
        dropped = carrying & (torch.norm(tcp_pos - target_cube, dim=-1) > cfg.drop_abort_dist)
        next_state = torch.where(dropped, torch.full_like(s, self.GO_ABOVE), next_state)

        next_state = torch.where(unreachable, torch.full_like(s, self.IDLE), next_state)

        self.timer = torch.where(next_state != s, torch.zeros_like(self.timer), self.timer)
        self.state = next_state

    """
    Introspection helpers for observations.
    """

    def tcp_pos_local(self) -> torch.Tensor:
        """Grasp-point position in the robot base frame, shape [num_envs, 3], [m]."""
        hand_pose_w = self.robot.data.body_pose_w.torch[:, self.hand_body_id]
        hand_pos = hand_pose_w[:, 0:3] - self._base_pos_w
        return hand_pos + quat_apply(
            hand_pose_w[:, 3:7], torch.tensor([0.0, 0.0, self.cfg.tcp_offset], device=self.device).repeat(self.num_envs, 1)
        )

    def gripper_opening(self) -> torch.Tensor:
        """Total finger opening, shape [num_envs], [m]."""
        return self.robot.data.joint_pos.torch[:, self.finger_joint_ids].sum(dim=-1)

    def at_read_pose(self) -> torch.Tensor:
        """Whether the hand is near the cue-read pose, shape [num_envs], bool."""
        hand_pose_w = self.robot.data.body_pose_w.torch[:, self.hand_body_id]
        hand_pos = hand_pose_w[:, 0:3] - self._base_pos_w
        return torch.norm(hand_pos - self._read_pos, dim=-1) < 0.35
