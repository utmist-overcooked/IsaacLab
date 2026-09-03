# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the scripted macro-action executor.

Kept import-light (no simulation-side modules) because task config modules are
imported before the simulation app starts.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass


@configclass
class MacroExecutorCfg:
    """Waypoint and timing parameters for the scripted executor.

    All positions are in the robot's base frame [m]. The base frame must share the
    world axes orientation (both stations spawn the robot with identity rotation).
    """

    tcp_offset: float = 0.1034
    """Distance from the ``panda_hand`` frame to the grasp point between the fingertips [m]."""

    approach_height: float = 0.24
    """Hand height for the pre-grasp waypoint above a cube [m]."""

    grasp_height_offset: float = 0.0
    """Vertical offset of the grasp target from the cube center [m]."""

    carry_height: float = 0.30
    """Hand height while transporting a cube [m]."""

    drop_height: float = 0.155
    """Hand height at which the cube is released over a receptacle [m]."""

    home_pos: tuple[float, float, float] = (0.45, 0.0, 0.40)
    """Idle hand position [m]."""

    read_pos: tuple[float, float, float] = (-0.15, -0.10, 0.45)
    """Hand position for reading the cue panel [m]. Behind the base, away from the table."""

    cue_panel_pos: tuple[float, float, float] = (-0.40, -0.30, 0.45)
    """Cue panel center, used to orient the hand during a read [m]."""

    pos_threshold: float = 0.03
    """Position error below which a motion state is considered reached [m]."""

    state_timeout_s: float = 1.6
    """Maximum time in a motion state before force-advancing [s]."""

    grasp_dwell_s: float = 0.35
    """Time to keep closing the gripper before lifting [s]."""

    release_dwell_s: float = 0.30
    """Time to keep the gripper open before retreating [s]."""

    gripper_open: float = 0.04
    """Finger joint position for an open gripper [m]."""

    gripper_close: float = 0.0
    """Finger joint position for a closed gripper [m]."""

    drop_abort_dist: float = 0.12
    """Distance between grasp point and cube beyond which a carry is considered dropped [m]."""
