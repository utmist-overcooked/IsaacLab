# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""COBOT two-Franka communication environment (direct MARL workflow)."""

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .cobot_env_cfg import BIN_POS, BOX_POS, CUBE_HALF_SIZE, CUBE_SPAWN_RANGE, STATION_Y, CobotEnvCfg
from .macro_executor import INSTR_NONE, INSTR_READ_CUE, FrankaMacroExecutor

# colors used when visual_cue_enabled paints displayColor (rgb in [0, 1])
CUBE_COLORS = {"red": (0.85, 0.08, 0.08), "blue": (0.08, 0.15, 0.85), "green": (0.08, 0.70, 0.12)}
PANEL_OFF_COLOR = (0.35, 0.35, 0.35)
TARGET_COLORS = ((0.85, 0.08, 0.08), (0.08, 0.15, 0.85))  # index by target id: 0=red, 1=blue

_STATIONS = ("a", "b")
_STATION_SIGNS = (1.0, -1.0)
_CUBE_ORDER = ("red", "blue", "green")  # indices 0/1 = receiver task, 2 = sender filler

# "Oracle" debug view: perched above and to one side of the divider wall so both stations
# are in frame at once. Env-local (eye, target); world == env-local at num_envs=1. Mirrors
# the ``oracle_debug`` entry in scripts/cobot/capture_views.py.
_ORACLE_CAM_EYE = (3.741, 0.115, 2.6)
_ORACLE_CAM_TARGET = (2.899, 0.1, 2.061)


def _look_at_quat(eye: tuple[float, float, float], target: tuple[float, float, float]):
    """Quaternion (w, x, y, z), "world" convention, pointing +X from ``eye`` at ``target``."""
    from isaaclab.utils.math import quat_from_euler_xyz

    dx, dy, dz = (t - c for c, t in zip(eye, target))
    yaw = torch.atan2(torch.tensor([dy]), torch.tensor([dx]))
    pitch = torch.atan2(torch.tensor([-dz]), torch.hypot(torch.tensor([dx]), torch.tensor([dy])))
    quat = quat_from_euler_xyz(torch.zeros(1), pitch, yaw)
    return tuple(quat[0].tolist())


class CobotEnv(DirectMARLEnv):
    """Two loosely-coupled Franka stations with an asymmetric-information color cue.

    Stations are asymmetric per episode: the receiver's table holds the red and blue
    cubes and the box (the target-color receptacle); the sender's table holds the green
    cube and the bin (the color-agnostic filler receptacle). Because the informed arm
    is (re)assigned per episode, the equipment is teleported to the matching station at
    reset. See :class:`CobotEnvCfg` for the task description and knobs.

    One env step is one macro tick: each agent submits a (macro-action, message) pair,
    the scripted executor runs the macro-action for ``decimation`` physics steps, and
    messages are delivered in the partner's next-tick observation.
    """

    cfg: CobotEnvCfg

    def __init__(self, cfg: CobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        n, dev = self.num_envs, self.device
        # world base positions per arm
        offs_a = torch.tensor([0.0, STATION_Y, 0.0], device=dev)
        offs_b = torch.tensor([0.0, -STATION_Y, 0.0], device=dev)
        self._base_pos_w = [self.scene.env_origins + offs_a, self.scene.env_origins + offs_b]

        # scripted executors, one per arm (the frozen VLA replaces these behind the same interface)
        self.executors = [
            FrankaMacroExecutor(
                cfg.executor, robot, self._base_pos_w[i], BOX_POS, BIN_POS, self.physics_dt, n, dev
            )
            for i, robot in enumerate(self._robots)
        ]

        # task state buffers
        self.target_color = torch.zeros(n, dtype=torch.long, device=dev)  # 0=red, 1=blue
        self.informed_arm = torch.zeros(n, dtype=torch.long, device=dev)  # 0=arm_a, 1=arm_b
        self.last_read = torch.zeros(n, 2, dtype=torch.long, device=dev)  # 0=never, 1=off, 2=red, 3=blue
        self.ticks_since_read = torch.zeros(n, 2, device=dev)
        self.macro_actions = torch.full((n, 2), INSTR_NONE, dtype=torch.long, device=dev)
        self.messages = torch.zeros(n, 2, dtype=torch.long, device=dev)  # emitted this tick, 0=null
        self.glimpse = torch.zeros(n, 2, dtype=torch.long, device=dev)  # 0=none, 1=red, 2=blue
        self.reward_buf = torch.zeros(n, device=dev)
        self.prev_base_obs = torch.zeros(n, 2, cfg.single_obs_dim, device=dev)

        # per-episode counters for logging
        self._ep = {
            key: torch.zeros(n, device=dev)
            for key in ("correct_places", "wrong_places", "green_places", "reads", "messages_sent", "flips")
        }

        # displayColor handles for runtime visuals
        self._panel_attrs: list = []  # one panel attr per env
        if self.cfg.visual_cue_enabled:
            self._setup_visuals()

    """
    Scene creation.
    """

    def _setup_scene(self):
        cfg = self.cfg
        # robots
        self._robots = [Articulation(cfg.robot_a_cfg), Articulation(cfg.robot_b_cfg)]

        # cubes: one red, one blue (receiver task), one green (sender filler) per env;
        # which table they sit on depends on the per-episode roles (teleported at reset)
        cube_props = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
        )
        self._cubes: list[RigidObject] = []
        for ci, color in enumerate(_CUBE_ORDER):
            spawn = sim_utils.CuboidCfg(
                size=(2 * CUBE_HALF_SIZE,) * 3,
                rigid_props=cube_props,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=None if cfg.visual_cue_enabled else sim_utils.PreviewSurfaceCfg(
                    diffuse_color=CUBE_COLORS[color]
                ),
            )
            obj_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Cube{color.capitalize()}",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45 + 0.07 * ci, STATION_Y, CUBE_HALF_SIZE)),
                spawn=spawn,
            )
            self._cubes.append(RigidObject(obj_cfg))

        # receptacle plates: kinematic rigid objects so they can follow the per-episode
        # role assignment (box -> receiver station, bin -> sender station)
        plate_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        self._box_plate = RigidObject(
            RigidObjectCfg(
                prim_path="/World/envs/env_.*/BoxPlate",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(BOX_POS[0], -STATION_Y + BOX_POS[1], 0.01)),
                spawn=sim_utils.CuboidCfg(
                    size=(0.16, 0.16, 0.02),
                    rigid_props=plate_props,
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.28)),
                ),
            )
        )
        self._bin_plate = RigidObject(
            RigidObjectCfg(
                prim_path="/World/envs/env_.*/BinPlate",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(BIN_POS[0], STATION_Y + BIN_POS[1], 0.01)),
                spawn=sim_utils.CuboidCfg(
                    size=(0.16, 0.16, 0.02),
                    rigid_props=plate_props,
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.30, 0.15)),
                ),
            )
        )

        # cue panel: a single kinematic slab per env that follows the sender's station
        # (teleported at reset), so the receiver's side has no panel at all
        panel_pos = self.cfg.executor.cue_panel_pos
        self._cue_panel = RigidObject(
            RigidObjectCfg(
                prim_path="/World/envs/env_.*/CuePanel",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(panel_pos[0], STATION_Y + panel_pos[1], panel_pos[2])
                ),
                spawn=sim_utils.CuboidCfg(
                    size=(0.02, 0.30, 0.30),
                    rigid_props=plate_props,
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
            )
        )

        # static station furniture under env_0 (cloned afterwards)
        table_cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        wall_cfg = sim_utils.CuboidCfg(
            size=(3.2, 0.06, 2.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.52)),
        )

        for st, sign in zip(_STATIONS, _STATION_SIGNS):
            table_cfg.func(
                f"/World/envs/env_0/Table{st.upper()}",
                table_cfg,
                translation=(0.5, sign * STATION_Y, 0.0),
                orientation=(0.0, 0.0, 0.7071068, 0.7071068),
            )
        wall_cfg.func("/World/envs/env_0/DividerWall", wall_cfg, translation=(0.5, 0.0, 0.2))

        # ground and light (shared)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # clone; USD copies (not references) when we need per-env displayColor writes
        self.scene.clone_environments(copy_from_source=self.cfg.visual_cue_enabled)

        # register with the scene
        self.scene.articulations["robot_a"] = self._robots[0]
        self.scene.articulations["robot_b"] = self._robots[1]
        for ci, color in enumerate(_CUBE_ORDER):
            self.scene.rigid_objects[f"cube_{color}"] = self._cubes[ci]
        self.scene.rigid_objects["box_plate"] = self._box_plate
        self.scene.rigid_objects["bin_plate"] = self._bin_plate
        self.scene.rigid_objects["cue_panel"] = self._cue_panel

        # optional cameras (for the pixel/occlusion phases)
        if self.cfg.enable_cameras:
            from isaaclab.sensors import Camera, CameraCfg

            w, h = self.cfg.camera_resolution
            pinhole = sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
            )
            for st, sign in zip(_STATIONS, _STATION_SIGNS):
                suffix = st.upper()
                wrist_cfg = CameraCfg(
                    prim_path=f"/World/envs/env_.*/Robot{suffix}/panda_hand/wrist_cam",
                    width=w,
                    height=h,
                    data_types=["rgb"],
                    spawn=pinhole,
                    offset=CameraCfg.OffsetCfg(
                        pos=(0.13, 0.0, -0.15), rot=(0.03701, 0.03701, -0.70614, -0.70614), convention="ros"
                    ),
                )
                station_cfg = CameraCfg(
                    prim_path=f"/World/envs/env_.*/StationCam{suffix}",
                    width=w,
                    height=h,
                    data_types=["rgb"],
                    spawn=pinhole,
                    offset=CameraCfg.OffsetCfg(
                        pos=(1.0, sign * STATION_Y, 0.4), rot=(-0.61237, -0.61237, 0.35355, 0.35355), convention="ros"
                    ),
                )
                self.scene.sensors[f"wrist_cam_{st}"] = Camera(wrist_cfg)
                self.scene.sensors[f"station_cam_{st}"] = Camera(station_cfg)

        # fixed "oracle" overview camera for debugging/demos only (never observed)
        if self.cfg.enable_debug_camera:
            from isaaclab.sensors import Camera, CameraCfg

            w, h = self.cfg.camera_resolution
            oracle_cfg = CameraCfg(
                prim_path="/World/envs/env_.*/OracleCam",
                width=w,
                height=h,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=_ORACLE_CAM_EYE, rot=_look_at_quat(_ORACLE_CAM_EYE, _ORACLE_CAM_TARGET), convention="world"
                ),
            )
            self.scene.sensors["oracle_cam"] = Camera(oracle_cfg)

    def _setup_visuals(self):
        """Cache displayColor attributes and paint the static cube colors."""
        from pxr import Usd, UsdGeom

        stage = self.sim.stage

        def color_attr(path: str):
            prim = stage.GetPrimAtPath(path)
            for p in Usd.PrimRange(prim):
                if p.IsA(UsdGeom.Gprim):
                    return UsdGeom.Gprim(p).GetDisplayColorAttr()
            raise RuntimeError(f"No gprim found under {path}")

        for i in range(self.num_envs):
            self._panel_attrs.append(color_attr(f"/World/envs/env_{i}/CuePanel"))
            for color in _CUBE_ORDER:
                color_attr(f"/World/envs/env_{i}/Cube{color.capitalize()}").Set([CUBE_COLORS[color]])
        # initialize panels to off
        for attr in self._panel_attrs:
            attr.Set([PANEL_OFF_COLOR])

    def _update_panel_visuals(self, env_ids: torch.Tensor):
        """Sync the (sender-side) panel color with the current target for the given envs."""
        if not self.cfg.visual_cue_enabled:
            return
        targets = self.target_color[env_ids].tolist()
        for env_id, tgt in zip(env_ids.tolist(), targets):
            self._panel_attrs[env_id].Set([TARGET_COLORS[tgt]])

    """
    Role helpers.
    """

    def _arm_base(self, arm_ids: torch.Tensor) -> torch.Tensor:
        """World base position of the given arm per env, shape [num_envs, 3], [m]."""
        base = torch.stack(self._base_pos_w, dim=1)  # [n, 2, 3]
        return base[torch.arange(self.num_envs, device=self.device), arm_ids]

    def _cube_owner(self, cube_idx: int) -> torch.Tensor:
        """Arm index that owns the cube this episode (receiver for red/blue, sender for green)."""
        return self.informed_arm if cube_idx == 2 else 1 - self.informed_arm

    """
    MDP hooks.
    """

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]):
        for i, agent in enumerate(self.cfg.possible_agents):
            act = actions[agent].long()
            self.macro_actions[:, i] = act[:, 0]
            self.messages[:, i] = act[:, 1]

        # knob 1: mid-episode target flips (only the informed side can perceive them)
        if self.cfg.flip_prob > 0.0:
            flip = torch.rand(self.num_envs, device=self.device) < self.cfg.flip_prob
            if flip.any():
                self.target_color = torch.where(flip, 1 - self.target_color, self.target_color)
                self._ep["flips"] += flip.float()
                self._update_panel_visuals(flip.nonzero(as_tuple=False).squeeze(-1))

        for i in range(2):
            self.executors[i].set_instructions(self.macro_actions[:, i])

        self._ep["messages_sent"] += (self.messages > 0).any(dim=-1).float()

    def _apply_action(self):
        for i in range(2):
            self.executors[i].step(self._cube_pos_local(i))

    def _cube_pos_local(self, arm: int) -> torch.Tensor:
        """All cube positions in the given arm's base frame, shape [num_envs, 3, 3], [m].

        Cubes on the partner's table simply come out far away; the executor treats
        out-of-workspace targets as unreachable and idles.
        """
        pos = torch.stack([cube.data.root_pose_w.torch[:, 0:3] for cube in self._cubes], dim=1)
        return pos - self._base_pos_w[arm].unsqueeze(1)

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._resolve_tick()
        time_out = self.episode_length_buf >= self.max_episode_length
        terminated = torch.zeros_like(time_out)
        return {a: terminated for a in self.cfg.possible_agents}, {a: time_out for a in self.cfg.possible_agents}

    def _resolve_tick(self):
        """End-of-tick bookkeeping: placements, cube respawns, cue reads, glimpses, reward."""
        cfg = self.cfg
        n, dev = self.num_envs, self.device
        ar = torch.arange(n, device=dev)
        self.reward_buf.zero_()

        box_c = torch.tensor(BOX_POS, device=dev)
        bin_c = torch.tensor(BIN_POS, device=dev)
        tcp_w = torch.stack(
            [ex.tcp_pos_local() + self._base_pos_w[i] for i, ex in enumerate(self.executors)], dim=1
        )  # [n, 2, 3]
        cube_pos_w = torch.stack([cube.data.root_pose_w.torch[:, 0:3] for cube in self._cubes], dim=1)  # [n, 3, 3]

        for ci in range(3):
            owner = self._cube_owner(ci)
            owner_base = self._arm_base(owner)
            local = cube_pos_w[:, ci] - owner_base  # cube in its owner station's frame
            held = torch.norm(cube_pos_w[:, ci] - tcp_w[ar, owner], dim=-1) < 0.10
            low = local[:, 2] < cfg.place_max_height
            recep_c = bin_c if ci == 2 else box_c
            placed = (torch.norm(local[:, 0:2] - recep_c, dim=-1) < cfg.place_radius) & low & ~held

            if ci == 2:  # sender filler: green in the bin
                self.reward_buf += placed.float() * cfg.green_place_reward
                self._ep["green_places"] += placed.float()
            else:  # receiver task: red/blue in the box, judged against the CURRENT target
                correct = placed & (self.target_color == ci)
                wrong = placed & (self.target_color != ci)
                self.reward_buf += correct.float() * cfg.correct_place_reward
                self.reward_buf += wrong.float() * cfg.wrong_place_reward
                self._ep["correct_places"] += correct.float()
                self._ep["wrong_places"] += wrong.float()

            ids = placed.nonzero(as_tuple=False).squeeze(-1)
            if len(ids) > 0:
                self._respawn_cube(ci, ids)

        # cue reads: READ_CUE spent the tick at the panel; only the informed panel carries info
        for i in range(2):
            read = self.macro_actions[:, i] == INSTR_READ_CUE
            if cfg.strict_read:
                read = read & self.executors[i].at_read_pose()
            informed_here = self.informed_arm == i
            val = torch.where(informed_here, 2 + self.target_color, torch.ones_like(self.target_color))
            self.last_read[:, i] = torch.where(read, val, self.last_read[:, i])
            self.ticks_since_read[:, i] = torch.where(read, torch.zeros(n, device=dev), self.ticks_since_read[:, i] + 1)
            self._ep["reads"] += read.float()

        # knob 2: receiver's independent glimpse of the target
        self.glimpse.zero_()
        receiver = 1 - self.informed_arm  # [n]
        if cfg.comm_mode == "oracle":
            glimpse_val = 1 + self.target_color
        elif cfg.asymmetry_p > 0.0:
            got = torch.rand(n, device=dev) < cfg.asymmetry_p
            glimpse_val = torch.where(got, 1 + self.target_color, torch.zeros_like(self.target_color))
        else:
            glimpse_val = torch.zeros_like(self.target_color)
        self.glimpse[ar, receiver] = glimpse_val

    def _respawn_cube(self, cube_idx: int, env_ids: torch.Tensor):
        """Teleport a placed cube back to a random collision-free pose on its owner's table."""
        m = len(env_ids)
        owner_base = self._arm_base(self._cube_owner(cube_idx))[env_ids]
        # positions of the other cubes in the owner's frame (far-table cubes never conflict)
        cube_pos_w = torch.stack([c.data.root_pose_w.torch[:, 0:3] for c in self._cubes], dim=1)[env_ids]
        others = [c for c in range(3) if c != cube_idx]
        other_xy = cube_pos_w[:, others, 0:2] - owner_base[:, 0:2].unsqueeze(1)

        lo = torch.tensor([CUBE_SPAWN_RANGE["x"][0], CUBE_SPAWN_RANGE["y"][0]], device=self.device)
        hi = torch.tensor([CUBE_SPAWN_RANGE["x"][1], CUBE_SPAWN_RANGE["y"][1]], device=self.device)
        cand = lo + (hi - lo) * torch.rand(m, 8, 2, device=self.device)
        dists = torch.norm(cand.unsqueeze(2) - other_xy.unsqueeze(1), dim=-1)
        valid = (dists.min(dim=-1).values > self.cfg.min_cube_separation).float()
        pick = torch.argmax(valid, dim=-1)  # first valid candidate (or 0 if none)
        xy = cand[torch.arange(m, device=self.device), pick]

        pose = torch.zeros(m, 7, device=self.device)
        pose[:, 0:2] = xy + owner_base[:, 0:2]
        pose[:, 2] = CUBE_HALF_SIZE + 0.002
        pose[:, 6] = 1.0  # identity quat (xyzw)
        cube = self._cubes[cube_idx]
        cube.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        cube.write_root_velocity_to_sim_index(root_velocity=torch.zeros(m, 6, device=self.device), env_ids=env_ids)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # team reward: both agents receive the same scalar
        return {agent: self.reward_buf.clone() for agent in self.cfg.possible_agents}

    def _get_observations(self) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        n, dev = self.num_envs, self.device
        k1 = cfg.codebook_size + 1

        base_obs = []
        for i in range(2):
            ex = self.executors[i]
            tcp = ex.tcp_pos_local()
            opening = ex.gripper_opening().unsqueeze(-1)
            cube_pos = self._cube_pos_local(i)

            dist = torch.norm(cube_pos - tcp.unsqueeze(1), dim=-1)  # [n, 3]
            held_cube = (dist < 0.06) & (opening < 0.055)
            held = torch.zeros(n, 4, device=dev)
            held[:, 0] = (~held_cube.any(dim=-1)).float()
            held[:, 1:4] = held_cube.float()

            instr = torch.zeros(n, 4, device=dev)
            valid = ex.instruction >= 0
            instr[valid] = torch.nn.functional.one_hot(ex.instruction[valid], 4).float()

            parts = [
                tcp,
                opening,
                held,
                cube_pos.reshape(n, 9),
                instr,
                ex.state.float().unsqueeze(-1) / 10.0,
                torch.nn.functional.one_hot(self.last_read[:, i], 4).float(),
            ]
            if cfg.observe_ticks_since_read:
                parts.append((self.ticks_since_read[:, i] / 20.0).clamp(max=2.0).unsqueeze(-1))
            parts.append(torch.nn.functional.one_hot(self.glimpse[:, i], 3).float())

            # incoming message slot (partner's emission this tick, delivered next tick by
            # construction: obs computed at the END of the step are the NEXT tick's input)
            msg = torch.zeros(n, k1, device=dev)
            j = 1 - i
            if cfg.comm_mode == "channel":
                msg = torch.nn.functional.one_hot(self.messages[:, j].clamp(max=k1 - 1), k1).float()
            elif cfg.comm_mode == "intent":
                pj = self.executors[j].instruction
                valid_j = pj >= 0
                msg[valid_j] = torch.nn.functional.one_hot(1 + pj[valid_j], k1).float()
            parts.append(msg)

            if cfg.observe_role:
                parts.append((self.informed_arm == i).float().unsqueeze(-1))
            parts.append((self.episode_length_buf.float() / self.max_episode_length).unsqueeze(-1))
            base_obs.append(torch.cat(parts, dim=-1))

        obs = {}
        for i, agent in enumerate(self.cfg.possible_agents):
            if cfg.comm_mode == "obs_broadcast":
                obs[agent] = torch.cat([base_obs[i], self.prev_base_obs[:, 1 - i]], dim=-1)
            else:
                obs[agent] = base_obs[i]
        self.prev_base_obs = torch.stack(base_obs, dim=1)
        return obs

    def _get_states(self) -> torch.Tensor:
        target = torch.nn.functional.one_hot(self.target_color, 2).float()
        return torch.cat(
            [
                self.obs_dict[self.cfg.possible_agents[0]],
                self.obs_dict[self.cfg.possible_agents[1]],
                target,
                self.informed_arm.float().unsqueeze(-1),
            ],
            dim=-1,
        )

    def _reset_idx(self, env_ids: Sequence[int]):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        # flush per-episode logs before clearing counters
        if len(env_ids) > 0:
            log = self.extras.setdefault("log", {})
            for key, buf in self._ep.items():
                log[f"Episode/{key}"] = buf[env_ids].mean().item()
                buf[env_ids] = 0.0

        super()._reset_idx(env_ids)

        # robots to default pose
        for robot in self._robots:
            joint_pos = robot.data.default_joint_pos.torch[env_ids]
            joint_vel = robot.data.default_joint_vel.torch[env_ids]
            robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
            robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
            robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)

        # task state
        self.target_color[env_ids] = (torch.rand(len(env_ids), device=self.device) < 0.5).long()
        if self.cfg.informed_arm is None:
            self.informed_arm[env_ids] = (torch.rand(len(env_ids), device=self.device) < 0.5).long()
        else:
            self.informed_arm[env_ids] = int(self.cfg.informed_arm)
        self.last_read[env_ids] = 0
        self.ticks_since_read[env_ids] = 0.0
        self.macro_actions[env_ids] = INSTR_NONE
        self.messages[env_ids] = 0
        self.glimpse[env_ids] = 0
        self.prev_base_obs[env_ids] = 0.0
        for ex in self.executors:
            ex.reset_idx(env_ids)

        # receptacles follow the roles: box to the receiver's table, bin to the sender's
        m = len(env_ids)
        recv_base = self._arm_base(1 - self.informed_arm)[env_ids]
        send_base = self._arm_base(self.informed_arm)[env_ids]
        panel_local = self.cfg.executor.cue_panel_pos
        movers = (
            (self._box_plate, recv_base, (BOX_POS[0], BOX_POS[1], 0.01)),
            (self._bin_plate, send_base, (BIN_POS[0], BIN_POS[1], 0.01)),
            (self._cue_panel, send_base, panel_local),
        )
        for obj, base, local in movers:
            pose = torch.zeros(m, 7, device=self.device)
            pose[:, 0] = base[:, 0] + local[0]
            pose[:, 1] = base[:, 1] + local[1]
            pose[:, 2] = local[2]
            pose[:, 6] = 1.0
            obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)

        # cubes follow the roles too: red/blue on the receiver's table, green on the sender's
        placed_rb: list[torch.Tensor] = []
        for ci in range(3):
            base = send_base if ci == 2 else recv_base
            xy = self._sample_spawn_xy(env_ids, [] if ci == 2 else placed_rb)
            if ci != 2:
                placed_rb.append(xy)
            pose = torch.zeros(m, 7, device=self.device)
            pose[:, 0:2] = xy + base[:, 0:2]
            pose[:, 2] = CUBE_HALF_SIZE + 0.002
            pose[:, 6] = 1.0
            cube = self._cubes[ci]
            cube.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
            cube.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(m, 6, device=self.device), env_ids=env_ids
            )

        self._update_panel_visuals(env_ids)

    def _sample_spawn_xy(self, env_ids: torch.Tensor, placed_xy: list[torch.Tensor]) -> torch.Tensor:
        """Sample station-local spawn xy for one cube per env, keeping min separation from
        already-placed cubes. Falls back to the first candidate if none satisfies it."""
        m = len(env_ids)
        lo = torch.tensor([CUBE_SPAWN_RANGE["x"][0], CUBE_SPAWN_RANGE["y"][0]], device=self.device)
        hi = torch.tensor([CUBE_SPAWN_RANGE["x"][1], CUBE_SPAWN_RANGE["y"][1]], device=self.device)
        cand = lo + (hi - lo) * torch.rand(m, 8, 2, device=self.device)
        if not placed_xy:
            return cand[:, 0]
        others = torch.stack(placed_xy, dim=1)  # [m, c, 2]
        dists = torch.norm(cand.unsqueeze(2) - others.unsqueeze(1), dim=-1)  # [m, 8, c]
        valid = (dists.min(dim=-1).values > self.cfg.min_cube_separation).float()
        pick = torch.argmax(valid, dim=-1)
        return cand[torch.arange(m, device=self.device), pick]
