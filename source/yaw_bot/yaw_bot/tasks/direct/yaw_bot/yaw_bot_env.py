# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.terrains import TerrainImporter

from .yaw_bot_env_cfg import YawBotEnvCfg


class YawBotEnv(DirectRLEnv):
    cfg: YawBotEnvCfg

    def __init__(self, cfg: YawBotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # joint indices
        self._left_hip_dof_idx, _ = self.robot.find_joints(self.cfg.left_hip_joint_name)
        self._left_knee_dof_idx, _ = self.robot.find_joints(self.cfg.left_knee_joint_name)
        self._left_wheel_dof_idx, _ = self.robot.find_joints(self.cfg.left_wheel_joint_name)
        self._right_hip_dof_idx, _ = self.robot.find_joints(self.cfg.right_hip_joint_name)
        self._right_knee_dof_idx, _ = self.robot.find_joints(self.cfg.right_knee_joint_name)
        self._right_wheel_dof_idx, _ = self.robot.find_joints(self.cfg.right_wheel_joint_name)
        self._body_link_ids, _ = self.robot.find_bodies(self.cfg.body_link_name)

        self._servo_joint_ids = torch.tensor(
            [
                self._left_hip_dof_idx[0],
                self._left_knee_dof_idx[0],
                self._right_hip_dof_idx[0],
                self._right_knee_dof_idx[0],
            ],
            device=self.device,
            dtype=torch.long,
        )

        self._wheel_joint_ids = torch.tensor(
            [
                self._left_wheel_dof_idx[0],
                self._right_wheel_dof_idx[0],
            ],
            device=self.device,
            dtype=torch.long,
        )

        self._all_joint_ids = torch.tensor(
            [
                self._left_hip_dof_idx[0],
                self._left_knee_dof_idx[0],
                self._left_wheel_dof_idx[0],
                self._right_hip_dof_idx[0],
                self._right_knee_dof_idx[0],
                self._right_wheel_dof_idx[0],
            ],
            device=self.device,
            dtype=torch.long,
        )

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.last_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._mapped_parallel_hip_targets = torch.zeros((self.num_envs, 2), device=self.device)
        self._disturbance_force = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._disturbance_torque = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._disturbance_steps_remaining = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._commands = torch.zeros((self.num_envs, 2), device=self.device)
        self._command_time_left = torch.zeros((self.num_envs,), device=self.device)
        self._command_dt = self.cfg.sim.dt * self.cfg.decimation

        # default servo pose is defined in (a, b) space and mapped into simulation joints.
        default_joint_pos = self.robot.data.default_joint_pos
        self._default_branch_hip_joint_pos = torch.full(
            (self.num_envs, 2),
            self.cfg.default_branch_hip_angle,
            device=self.device,
            dtype=default_joint_pos.dtype,
        )
        self._default_mapped_parallel_hip_pos = torch.full(
            (self.num_envs, 2),
            self.cfg.default_mapped_hip_angle,
            device=self.device,
            dtype=default_joint_pos.dtype,
        )
        self._default_servo_joint_pos = self._map_branch_and_parallel_hips_to_sim_servo_targets(
            self._default_branch_hip_joint_pos,
            self._default_mapped_parallel_hip_pos,
        )
        self._default_knee_joint_pos = self._default_servo_joint_pos[:, [1, 3]].clone()

        # per-step commanded servo targets
        self._servo_position_targets = self._default_servo_joint_pos.clone()
        self._wheel_effort_targets = torch.zeros((self.num_envs, 2), device=self.device)

        # sign mapping for servo semantic consistency
        # left hip a, left mapped hip b, right hip a, right mapped hip b
        self._servo_action_sign = torch.tensor(
            [
                1.0,   # left hip
                1.0,   # left mapped hip
                1.0,   # right hip
                1.0,   # right mapped hip
            ],
            device=self.device,
            dtype=torch.float,
        ).unsqueeze(0)

        # sign mapping for wheels so same positive action means forward motion
        # left wheel, right wheel
        self._wheel_action_sign = torch.tensor(
            [
                1.0,   # left wheel
                -1.0,  # right wheel: mirrored axis, so flip sign
            ],
            device=self.device,
            dtype=torch.float,
        ).unsqueeze(0)
        # joint limits for clamping servo targets
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits
        self._servo_lower_limits = soft_joint_pos_limits[:, self._servo_joint_ids, 0]
        self._servo_upper_limits = soft_joint_pos_limits[:, self._servo_joint_ids, 1]
        self._branch_hip_lower_limits = self._servo_lower_limits[:, [0, 2]]
        self._branch_hip_upper_limits = self._servo_upper_limits[:, [0, 2]]
        self._mapped_parallel_hip_lower_limits = torch.full(
            (self.num_envs, 2),
            self.cfg.mapped_hip_lower_limit,
            device=self.device,
        )
        self._mapped_parallel_hip_upper_limits = torch.full(
            (self.num_envs, 2),
            self.cfg.mapped_hip_upper_limit,
            device=self.device,
        )
        wheel_effort_limit = float(self.cfg.robot_cfg.actuators["wheel_joints"].effort_limit)
        self._wheel_effort_lower_limits = torch.full((self.num_envs, 2), -wheel_effort_limit, device=self.device)
        self._wheel_effort_upper_limits = torch.full((self.num_envs, 2), wheel_effort_limit, device=self.device)

    def _map_normalized_actions_to_range(
        self, actions: torch.Tensor, lower_limits: torch.Tensor, upper_limits: torch.Tensor
    ) -> torch.Tensor:
        """Map normalized actions in [-1, 1] to the full closed interval [lower, upper]."""
        midpoint = 0.5 * (upper_limits + lower_limits)
        half_range = 0.5 * (upper_limits - lower_limits)
        return midpoint + actions * half_range

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        """Resample forward and yaw velocity commands."""
        if env_ids.numel() == 0:
            return

        lin_low, lin_high = self.cfg.command_lin_vel_x_range
        yaw_low, yaw_high = self.cfg.command_yaw_vel_range
        time_low, time_high = self.cfg.command_resample_time_range

        lin_cmd = self._sample_command_with_deadzone(
            env_ids.numel(),
            lin_low,
            lin_high,
            self.cfg.command_lin_vel_x_min_abs,
        )
        yaw_cmd = torch.zeros(env_ids.numel(), device=self.device)
        yaw_active = torch.rand(env_ids.numel(), device=self.device) < self.cfg.command_yaw_probability
        if torch.any(yaw_active):
            yaw_cmd[yaw_active] = self._sample_command_with_deadzone(
                int(torch.sum(yaw_active).item()),
                yaw_low,
                yaw_high,
                self.cfg.command_yaw_vel_min_abs,
            )

        self._commands[env_ids, 0] = lin_cmd
        self._commands[env_ids, 1] = yaw_cmd
        self._command_time_left[env_ids] = time_low + (time_high - time_low) * torch.rand(
            env_ids.numel(), device=self.device
        )

    def _sample_command_with_deadzone(
        self, num_samples: int, low: float, high: float, min_abs: float
    ) -> torch.Tensor:
        """Sample commands while avoiding a dead-zone around zero when requested."""
        if num_samples == 0:
            return torch.zeros(0, device=self.device)
        if min_abs <= 0.0 or not (low < 0.0 < high):
            return low + (high - low) * torch.rand(num_samples, device=self.device)

        samples = torch.empty(num_samples, device=self.device)
        choose_positive = torch.rand(num_samples, device=self.device) < 0.5

        pos_low = max(min_abs, 0.0)
        pos_high = high
        neg_low = low
        neg_high = min(-min_abs, 0.0)

        if pos_low < pos_high:
            pos_count = int(torch.sum(choose_positive).item())
            if pos_count > 0:
                samples[choose_positive] = pos_low + (pos_high - pos_low) * torch.rand(pos_count, device=self.device)
        else:
            choose_positive[:] = False

        choose_negative = ~choose_positive
        if neg_low < neg_high:
            neg_count = int(torch.sum(choose_negative).item())
            if neg_count > 0:
                samples[choose_negative] = neg_low + (neg_high - neg_low) * torch.rand(neg_count, device=self.device)
        else:
            samples[choose_negative] = pos_low + (pos_high - pos_low) * torch.rand(
                int(torch.sum(choose_negative).item()), device=self.device
            )

        return samples

    def _compute_equivalent_knee_angle_from_branch_hips(
        self, branch_hip_angles_deg: torch.Tensor, mapped_hip_angles_deg: torch.Tensor
    ) -> torch.Tensor:
        """Compute equivalent knee angle t from branch hip angle a and mapped hip angle b.

        All angles are in degrees. This implements the provided closed-form relationship but
        uses atan2 for stable quadrant handling:

            t = -180 + a + atan2(y, x) + acos(-sqrt(x^2 + y^2) / 200)

        where:
            x = 60 * (cos(b) - cos(a))
            y = 60 * (sin(b) + sin(a)) + 45
        """
        a_rad = torch.deg2rad(branch_hip_angles_deg)
        b_rad = torch.deg2rad(mapped_hip_angles_deg)
        x = 60.0 * (torch.cos(b_rad) - torch.cos(a_rad))
        y = 60.0 * (torch.sin(b_rad) + torch.sin(a_rad)) + 45.0
        chord = torch.sqrt(torch.square(x) + torch.square(y))
        inner_angle_deg = torch.rad2deg(torch.atan2(y, x))
        knee_triangle_angle_deg = torch.rad2deg(torch.arccos(torch.clamp(-chord / 200.0, -1.0, 1.0)))
        return -180.0 + branch_hip_angles_deg + inner_angle_deg + knee_triangle_angle_deg

    def _compute_equivalent_knee_angle_from_branch_hips_rad(
        self, branch_hip_angles_rad: torch.Tensor, mapped_hip_angles_rad: torch.Tensor
    ) -> torch.Tensor:
        """Same as `_compute_equivalent_knee_angle_from_branch_hips` but with radian inputs/outputs."""
        knee_deg = self._compute_equivalent_knee_angle_from_branch_hips(
            torch.rad2deg(branch_hip_angles_rad),
            torch.rad2deg(mapped_hip_angles_rad),
        )
        return torch.deg2rad(knee_deg)

    def _solve_mapped_hip_angle_from_equivalent_knee_angle(
        self, branch_hip_angles_rad: torch.Tensor, knee_angles_rad: torch.Tensor
    ) -> torch.Tensor:
        """Solve mapped hip angle b from branch hip angle a and equivalent knee angle t.

        This uses vectorized bisection on the hip joint limits. If the target does not bracket
        within the search range, we fall back to the branch hip angle for that leg.
        """
        lower = self._mapped_parallel_hip_lower_limits.clone()
        upper = self._mapped_parallel_hip_upper_limits.clone()

        f_lower = (
            self._compute_equivalent_knee_angle_from_branch_hips_rad(branch_hip_angles_rad, lower) - knee_angles_rad
        )
        f_upper = (
            self._compute_equivalent_knee_angle_from_branch_hips_rad(branch_hip_angles_rad, upper) - knee_angles_rad
        )

        has_bracket = f_lower * f_upper <= 0.0
        solution = branch_hip_angles_rad.clone()

        if torch.any(has_bracket):
            lo = lower.clone()
            hi = upper.clone()
            flo = f_lower.clone()
            for _ in range(28):
                mid = 0.5 * (lo + hi)
                f_mid = (
                    self._compute_equivalent_knee_angle_from_branch_hips_rad(branch_hip_angles_rad, mid)
                    - knee_angles_rad
                )
                choose_lower_half = flo * f_mid <= 0.0
                hi = torch.where(choose_lower_half, mid, hi)
                lo = torch.where(choose_lower_half, lo, mid)
                flo = torch.where(choose_lower_half, flo, f_mid)
            bisection_result = 0.5 * (lo + hi)
            solution = torch.where(has_bracket, bisection_result, solution)

        return solution

    def _map_branch_and_parallel_hips_to_sim_servo_targets(
        self, branch_hip_targets: torch.Tensor, mapped_parallel_hip_targets: torch.Tensor
    ) -> torch.Tensor:
        """Map branch hip angle a and parallel hip angle b to simulation servo targets [hip, knee, hip, knee]."""
        semantic_knee_targets = self._compute_equivalent_knee_angle_from_branch_hips_rad(
            branch_hip_targets,
            mapped_parallel_hip_targets,
        )
        servo_targets = torch.zeros(
            (branch_hip_targets.shape[0], 4),
            device=branch_hip_targets.device,
            dtype=branch_hip_targets.dtype,
        )
        servo_targets[:, [0, 2]] = branch_hip_targets
        servo_targets[:, 1] = semantic_knee_targets[:, 0]
        servo_targets[:, 3] = -semantic_knee_targets[:, 1]
        return servo_targets

    def _map_equivalent_knee_torques_to_sim_knee_torques(self, knee_torques: torch.Tensor) -> torch.Tensor:
        """Map equivalent knee torques to the simplified simulation knee joints.

        This is where the virtual-work-consistent torque mapping should live later.
        For now we keep the identity mapping so the equivalent knee torque is passed through:
            f(tau_knee_eq) = tau_knee_eq

        Args:
            knee_torques: Tensor of shape (num_envs, 2) ordered as [left_knee, right_knee].

        Returns:
            Tensor of shape (num_envs, 2) containing the simulation knee torques.
        """
        return knee_torques.clone()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.body_contact_sensor = ContactSensor(self.cfg.body_contact_sensor)
        terrain_cfg = self.cfg.rough_terrain if self.cfg.use_rough_terrain else self.cfg.flat_terrain
        terrain_cfg.num_envs = self.scene.cfg.num_envs
        terrain_cfg.env_spacing = self.scene.cfg.env_spacing
        self._terrain = TerrainImporter(terrain_cfg)
        self._terrain_has_curriculum = self.cfg.use_rough_terrain and self._terrain.terrain_origins is not None

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[terrain_cfg.prim_path])

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["body_contact"] = self.body_contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

        if self.cfg.use_velocity_commands and self.cfg.resample_commands:
            self._command_time_left -= self._command_dt
            resample_env_ids = torch.nonzero(self._command_time_left <= 0.0, as_tuple=False).squeeze(-1)
            if resample_env_ids.numel() > 0:
                self._resample_commands(resample_env_ids)

        if not (self.cfg.enable_disturbance_force or self.cfg.enable_disturbance_torque):
            self._disturbance_steps_remaining.zero_()
            self._disturbance_force.zero_()
            self._disturbance_torque.zero_()
            return

        # Random short wrench pulses make the policy robust to unexpected pushes.
        active_mask = self._disturbance_steps_remaining > 0
        self._disturbance_steps_remaining[active_mask] -= 1

        self._disturbance_force.zero_()
        self._disturbance_torque.zero_()

        ready_mask = self._disturbance_steps_remaining == 0
        force_trigger_mask = ready_mask & (
            torch.rand(self.num_envs, device=self.device) < self.cfg.disturbance_force_probability
        )
        force_trigger_env_ids = torch.nonzero(force_trigger_mask, as_tuple=False).squeeze(-1)
        if self.cfg.enable_disturbance_force and force_trigger_env_ids.numel() > 0:
            force = torch.zeros((force_trigger_env_ids.numel(), 1, 3), device=self.device)
            force[:, 0, :2] = 2.0 * torch.rand((force_trigger_env_ids.numel(), 2), device=self.device) - 1.0
            force *= self.cfg.disturbance_force_scale
            self._disturbance_force[force_trigger_env_ids] = force
            self._disturbance_steps_remaining[force_trigger_env_ids] = self.cfg.disturbance_force_duration_steps

        torque_trigger_mask = ready_mask & (
            torch.rand(self.num_envs, device=self.device) < self.cfg.disturbance_torque_probability
        )
        torque_trigger_env_ids = torch.nonzero(torque_trigger_mask, as_tuple=False).squeeze(-1)
        if self.cfg.enable_disturbance_torque and torque_trigger_env_ids.numel() > 0:
            torque = 2.0 * torch.rand((torque_trigger_env_ids.numel(), 1, 3), device=self.device) - 1.0
            torque *= self.cfg.disturbance_torque_scale
            self._disturbance_torque[torque_trigger_env_ids] = torque
            self._disturbance_steps_remaining[torque_trigger_env_ids] = torch.maximum(
                self._disturbance_steps_remaining[torque_trigger_env_ids],
                torch.full(
                    (torque_trigger_env_ids.numel(),),
                    self.cfg.disturbance_torque_duration_steps,
                    device=self.device,
                    dtype=self._disturbance_steps_remaining.dtype,
                ),
            )

        still_active_mask = self._disturbance_steps_remaining > 0
        self._disturbance_force[~still_active_mask] = 0.0
        self._disturbance_torque[~still_active_mask] = 0.0

    def _apply_action(self) -> None:
        # -----------------------------
        # 1) servo joints: position control
        # actions[0:4] -> [left hip a, left mapped hip b, right hip a, right mapped hip b]
        # -----------------------------
        servo_actions = self.actions[:, 0:4] * self._servo_action_sign
        branch_hip_targets = self._map_normalized_actions_to_range(
            servo_actions[:, [0, 2]],
            self._branch_hip_lower_limits,
            self._branch_hip_upper_limits,
        )
        self._mapped_parallel_hip_targets = self._map_normalized_actions_to_range(
            servo_actions[:, [1, 3]],
            self._mapped_parallel_hip_lower_limits,
            self._mapped_parallel_hip_upper_limits,
        )
        servo_targets = self._map_branch_and_parallel_hips_to_sim_servo_targets(
            branch_hip_targets,
            self._mapped_parallel_hip_targets,
        )
        self._servo_position_targets = servo_targets

        self.robot.set_joint_position_target(
            self._servo_position_targets,
            joint_ids=self._servo_joint_ids,
        )

        # -----------------------------
        # 2) wheel joints: torque control
        # actions[4:6] -> wheel torques
        # same positive action = robot forward
        # -----------------------------
        wheel_actions = self.actions[:, 4:6] * self._wheel_action_sign
        wheel_efforts = self._map_normalized_actions_to_range(
            wheel_actions,
            self._wheel_effort_lower_limits,
            self._wheel_effort_upper_limits,
        )
        self._wheel_effort_targets = wheel_efforts

        self.robot.set_joint_effort_target(
            wheel_efforts,
            joint_ids=self._wheel_joint_ids,
        )
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=self._disturbance_force,
            torques=self._disturbance_torque,
            body_ids=self._body_link_ids,
            is_global=True,
        )

    def _get_observations(self) -> dict:
        wheel_pos = self.robot.data.joint_pos[:, self._wheel_joint_ids]
        wheel_vel = self.robot.data.joint_vel[:, self._wheel_joint_ids]

        root_quat = self.robot.data.root_quat_w
        root_ang_vel = self.robot.data.root_ang_vel_w
        projected_gravity = self.robot.data.projected_gravity_b

        if self.cfg.enable_imu_noise:
            imu_quat = root_quat + torch.randn_like(root_quat) * self.cfg.imu_quat_noise_std
            imu_quat = torch.nn.functional.normalize(imu_quat, dim=-1)
            imu_ang_vel = root_ang_vel + torch.randn_like(root_ang_vel) * self.cfg.imu_ang_vel_noise_std
            imu_projected_gravity = (
                projected_gravity
                + torch.randn_like(projected_gravity) * self.cfg.imu_projected_gravity_noise_std
            )
        else:
            imu_quat = root_quat
            imu_ang_vel = root_ang_vel
            imu_projected_gravity = projected_gravity

        obs_parts = [
            imu_quat,               # 4
            imu_ang_vel,            # 3
            imu_projected_gravity,  # 3
        ]
        if self.cfg.use_velocity_commands:
            obs_parts.append(self._commands)  # 2
        obs_parts.extend(
            [
                wheel_pos,          # 2
                wheel_vel,          # 2
                self.last_actions,  # 6
            ]
        )
        obs = torch.cat(obs_parts, dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self.extras["log"] = {}
        servo_joint_pos = self.robot.data.joint_pos[:, self._servo_joint_ids]
        servo_joint_vel = self.robot.data.joint_vel[:, self._servo_joint_ids]
        projected_gravity = self.robot.data.projected_gravity_b
        root_lin_vel = self.robot.data.root_lin_vel_b
        root_ang_vel = self.robot.data.root_ang_vel_w
        wheel_vel = self.robot.data.joint_vel[:, self._wheel_joint_ids]

        gravity_xy_error = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
        servo_pose_error = torch.sum(torch.square(servo_joint_pos - self._default_servo_joint_pos), dim=1)

        rew_alive = self.cfg.rew_scale_alive * (1.0 - self.reset_terminated.float())
        rew_termination = self.cfg.rew_scale_terminated * self.reset_terminated.float()
        rew_angle = self.cfg.rew_scale_angle * gravity_xy_error
        rew_ang_vel = self.cfg.rew_scale_ang_vel * torch.sum(torch.square(root_ang_vel[:, :2]), dim=1)
        rew_vertical_vel = self.cfg.rew_scale_vertical_vel * torch.square(root_lin_vel[:, 2])
        rew_servo_joint_vel = self.cfg.rew_scale_servo_joint_vel * torch.sum(torch.square(servo_joint_vel), dim=1)
        rew_action_rate = self.cfg.rew_scale_action_rate * torch.sum(
            torch.square(self.actions - self.last_actions), dim=1
        )
        if self.cfg.use_velocity_commands:
            lin_vel_error = torch.square(root_lin_vel[:, 0] - self._commands[:, 0])
            yaw_vel_error = torch.square(root_ang_vel[:, 2] - self._commands[:, 1])
            # When a motion command is present, loosen "stand still" regularizers so they do not fight locomotion.
            command_motion_mag = torch.abs(self._commands[:, 0]) + 0.5 * torch.abs(self._commands[:, 1])
            standstill_weight = torch.exp(-command_motion_mag / 0.25)
            turn_stability_weight = torch.exp(-torch.abs(self._commands[:, 1]) / 0.25)
            rew_yaw_ang_vel = self.cfg.rew_scale_yaw_ang_vel * turn_stability_weight * torch.square(root_ang_vel[:, 2])
            rew_servo_pose = self.cfg.rew_scale_servo_pose * standstill_weight * servo_pose_error
            lin_track_score = torch.exp(-lin_vel_error / self.cfg.command_tracking_sigma_lin)
            lin_standstill_baseline = torch.exp(
                -torch.square(self._commands[:, 0]) / self.cfg.command_tracking_sigma_lin
            )
            yaw_track_score = torch.exp(-yaw_vel_error / self.cfg.command_tracking_sigma_yaw)
            yaw_standstill_baseline = torch.exp(
                -torch.square(self._commands[:, 1]) / self.cfg.command_tracking_sigma_yaw
            )
            # Do not pay out motion rewards while the robot is obviously pitching/falling.
            upright_motion_weight = torch.exp(
                -gravity_xy_error / self.cfg.command_tracking_upright_sigma
            ) * torch.exp(
                -torch.sum(torch.square(root_ang_vel[:, :2]), dim=1) / self.cfg.command_tracking_stability_sigma
            )
            rew_track_lin_vel = self.cfg.rew_scale_track_lin_vel * (
                lin_track_score - lin_standstill_baseline
            ) * upright_motion_weight
            rew_track_yaw_vel = self.cfg.rew_scale_track_yaw_vel * (
                yaw_track_score - yaw_standstill_baseline
            ) * upright_motion_weight
        else:
            rew_yaw_ang_vel = self.cfg.rew_scale_yaw_ang_vel * torch.square(root_ang_vel[:, 2])
            rew_servo_pose = self.cfg.rew_scale_servo_pose * servo_pose_error
            rew_track_lin_vel = torch.zeros(self.num_envs, device=self.device)
            rew_track_yaw_vel = torch.zeros(self.num_envs, device=self.device)
            turn_stability_weight = torch.ones(self.num_envs, device=self.device)

        # -----------------------------
        # leg symmetry penalty
        # -----------------------------
        left_hip = self.robot.data.joint_pos[:, self._left_hip_dof_idx[0]]
        left_knee = self.robot.data.joint_pos[:, self._left_knee_dof_idx[0]]
        right_hip = self.robot.data.joint_pos[:, self._right_hip_dof_idx[0]]
        right_knee = self.robot.data.joint_pos[:, self._right_knee_dof_idx[0]]

        # hips: same semantic direction
        hip_sym_error = torch.square(left_hip - right_hip)

        # knees: right knee axis is mirrored, so symmetric posture is left_knee + right_knee ~= 0
        knee_sym_error = torch.square(left_knee + right_knee)

        rew_leg_symmetry = self.cfg.rew_scale_leg_symmetry * turn_stability_weight * (hip_sym_error + knee_sym_error)

        # reward semantic wheel spinning so the policy is encouraged to actually drive.
        semantic_wheel_vel = wheel_vel * self._wheel_action_sign
        wheel_spin_reward = torch.mean(torch.abs(semantic_wheel_vel), dim=1)
        rew_wheel_spin = self.cfg.rew_scale_wheel_spin * wheel_spin_reward

        if self.cfg.use_velocity_commands:
            semantic_wheel_forward_vel = torch.mean(semantic_wheel_vel, dim=1)
            semantic_wheel_yaw_vel = 0.5 * (semantic_wheel_vel[:, 0] - semantic_wheel_vel[:, 1])
            rew_track_wheel_lin = self.cfg.rew_scale_track_wheel_lin * torch.tanh(
                3.0 * self._commands[:, 0] * semantic_wheel_forward_vel
            ) * upright_motion_weight
            rew_track_wheel_yaw = self.cfg.rew_scale_track_wheel_yaw * torch.tanh(
                2.0 * self._commands[:, 1] * semantic_wheel_yaw_vel
            ) * upright_motion_weight
        else:
            rew_track_wheel_lin = torch.zeros(self.num_envs, device=self.device)
            rew_track_wheel_yaw = torch.zeros(self.num_envs, device=self.device)

        # wheels should spin at similar semantic speeds during stationary balancing;
        # a persistent mismatch usually manifests as slow self-rotation.
        wheel_speed_diff = torch.square(wheel_vel[:, 0] + wheel_vel[:, 1])
        rew_wheel_speed_diff = self.cfg.rew_scale_wheel_speed_diff * turn_stability_weight * wheel_speed_diff

        # per-step diagnostics for wheel behavior
        log = self.extras["log"]
        log["Diagnostics/wheel_vel_left"] = torch.mean(wheel_vel[:, 0]).item()
        log["Diagnostics/wheel_vel_right"] = torch.mean(wheel_vel[:, 1]).item()
        log["Diagnostics/wheel_semantic_vel_mean"] = torch.mean(semantic_wheel_vel).item()
        log["Diagnostics/wheel_semantic_forward_vel"] = torch.mean(torch.mean(semantic_wheel_vel, dim=1)).item()
        log["Diagnostics/wheel_speed_diff_sq"] = torch.mean(wheel_speed_diff).item()
        log["Diagnostics/root_lin_vel_x"] = torch.mean(root_lin_vel[:, 0]).item()
        log["Diagnostics/root_ang_vel_z"] = torch.mean(root_ang_vel[:, 2]).item()
        log["Diagnostics/wheel_action_abs"] = torch.mean(torch.abs(self.actions[:, 4:6])).item()
        log["Diagnostics/wheel_effort_cmd_abs"] = torch.mean(torch.abs(self._wheel_effort_targets)).item()
        log["Diagnostics/servo_pose_error"] = torch.mean(servo_pose_error).item()
        log["Diagnostics/servo_joint_vel_sq"] = torch.mean(torch.sum(torch.square(servo_joint_vel), dim=1)).item()
        log["Diagnostics/gravity_xy_error"] = torch.mean(gravity_xy_error).item()
        log["Diagnostics/root_vertical_vel_abs"] = torch.mean(torch.abs(root_lin_vel[:, 2])).item()

        wheel_radius = self.cfg.wheel_radius
        wheel_forward_surface_speed = wheel_radius * torch.mean(semantic_wheel_vel, dim=1)
        slip_error = wheel_forward_surface_speed - root_lin_vel[:, 0]
        log["Diagnostics/wheel_surface_speed"] = torch.mean(wheel_forward_surface_speed).item()
        log["Diagnostics/wheel_surface_speed_abs"] = torch.mean(torch.abs(wheel_forward_surface_speed)).item()
        log["Diagnostics/wheel_body_speed_slip"] = torch.mean(slip_error).item()
        log["Diagnostics/wheel_body_speed_slip_abs"] = torch.mean(torch.abs(slip_error)).item()

        if self.cfg.use_velocity_commands:
            lin_pos_mask = self._commands[:, 0] > 0.05
            lin_neg_mask = self._commands[:, 0] < -0.05
            cmd_match = torch.sign(self._commands[:, 0]) * torch.sign(root_lin_vel[:, 0])
            active_lin_mask = torch.abs(self._commands[:, 0]) > 0.05
            if torch.any(active_lin_mask):
                log["Diagnostics/lin_cmd_sign_match_rate"] = torch.mean(
                    (cmd_match[active_lin_mask] > 0).float()
                ).item()
                log["Diagnostics/lin_cmd_response_mag"] = torch.mean(
                    torch.abs(root_lin_vel[active_lin_mask, 0])
                ).item()
            if torch.any(lin_pos_mask):
                log["Diagnostics/forward_cmd_root_lin_vel_x"] = torch.mean(root_lin_vel[lin_pos_mask, 0]).item()
                log["Diagnostics/forward_cmd_wheel_semantic_vel"] = torch.mean(
                    torch.mean(semantic_wheel_vel[lin_pos_mask], dim=1)
                ).item()
                log["Diagnostics/forward_cmd_success_rate"] = torch.mean(
                    (root_lin_vel[lin_pos_mask, 0] > 0.02).float()
                ).item()
                log["Diagnostics/forward_cmd_slip_abs"] = torch.mean(torch.abs(slip_error[lin_pos_mask])).item()
            if torch.any(lin_neg_mask):
                log["Diagnostics/backward_cmd_root_lin_vel_x"] = torch.mean(root_lin_vel[lin_neg_mask, 0]).item()
                log["Diagnostics/backward_cmd_wheel_semantic_vel"] = torch.mean(
                    torch.mean(semantic_wheel_vel[lin_neg_mask], dim=1)
                ).item()
                log["Diagnostics/backward_cmd_success_rate"] = torch.mean(
                    (root_lin_vel[lin_neg_mask, 0] < -0.02).float()
                ).item()
                log["Diagnostics/backward_cmd_slip_abs"] = torch.mean(torch.abs(slip_error[lin_neg_mask])).item()

        total_reward = (
            rew_alive
            + rew_termination
            + rew_angle
            + rew_ang_vel
            + rew_vertical_vel
            + rew_yaw_ang_vel
            + rew_servo_joint_vel
            + rew_action_rate
            + rew_servo_pose
            + rew_track_lin_vel
            + rew_track_yaw_vel
            + rew_track_wheel_lin
            + rew_track_wheel_yaw
            + rew_leg_symmetry
            + rew_wheel_spin
            + rew_wheel_speed_diff
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.disable_termination:
            false_dones = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            return false_dones, false_dones

        contact_force_norm = torch.norm(self.body_contact_sensor.data.net_forces_w, dim=-1)
        non_wheel_body_contact = torch.any(
            contact_force_norm > self.cfg.termination_contact_force_threshold,
            dim=1,
        )

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return non_wheel_body_contact, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if self._terrain_has_curriculum and env_ids.numel() > 0:
            episode_progress = self.episode_length_buf[env_ids].float() / float(self.max_episode_length)
            move_up = episode_progress >= self.cfg.terrain_level_up_threshold
            move_down = (episode_progress < self.cfg.terrain_level_down_threshold) & self.reset_terminated[env_ids]
            self._terrain.update_env_origins(env_ids, move_up, move_down)

        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        joint_pos[:, self._servo_joint_ids] = self._default_servo_joint_pos[env_ids]

        joint_vel[:, self._all_joint_ids] = 0.0

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 2] += self.cfg.reset_height_offset
        default_root_state[:, 7:13] = 0.0

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self._servo_position_targets[env_ids] = self._default_servo_joint_pos[env_ids]
        self._wheel_effort_targets[env_ids] = 0.0
        self._mapped_parallel_hip_targets[env_ids] = 0.0
        self._disturbance_force[env_ids] = 0.0
        self._disturbance_torque[env_ids] = 0.0
        self._disturbance_steps_remaining[env_ids] = 0
        if self.cfg.use_velocity_commands and self.cfg.resample_commands:
            self._resample_commands(env_ids)
        else:
            self._commands[env_ids] = 0.0
            self._command_time_left[env_ids] = 0.0

    def _post_physics_step(self):
        self.last_actions.copy_(self.actions)
