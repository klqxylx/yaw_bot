# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

from yaw_bot.robots import YAW_BOT_CFG


@configclass
class YawBotEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    disable_termination = False

    # spaces
    action_space = 6
    observation_space = 22
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # robot
    robot_cfg: ArticulationCfg = YAW_BOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=0.5,
        replicate_physics=True,
    )
    use_rough_terrain = False

    rough_terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG.replace(
            size=(1.2, 1.2),
            border_width=0.25,
            horizontal_scale=0.02,
            vertical_scale=0.001,
            curriculum=True,
            num_rows=8,
            num_cols=8,
            sub_terrains={
                "pyramid_stairs": ROUGH_TERRAINS_CFG.sub_terrains["pyramid_stairs"].replace(
                    step_height_range=(0.006, 0.03),
                    step_width=0.08,
                    platform_width=0.45,
                    border_width=0.08,
                ),
                "pyramid_stairs_inv": ROUGH_TERRAINS_CFG.sub_terrains["pyramid_stairs_inv"].replace(
                    step_height_range=(0.006, 0.03),
                    step_width=0.08,
                    platform_width=0.45,
                    border_width=0.08,
                ),
                "boxes": ROUGH_TERRAINS_CFG.sub_terrains["boxes"].replace(
                    grid_width=0.10,
                    grid_height_range=(0.006, 0.025),
                    platform_width=0.40,
                ),
                "random_rough": ROUGH_TERRAINS_CFG.sub_terrains["random_rough"].replace(
                    noise_range=(0.003, 0.02),
                    noise_step=0.003,
                    border_width=0.05,
                ),
                "hf_pyramid_slope": ROUGH_TERRAINS_CFG.sub_terrains["hf_pyramid_slope"].replace(
                    slope_range=(0.0, 0.18),
                    platform_width=0.40,
                    border_width=0.05,
                ),
                "hf_pyramid_slope_inv": ROUGH_TERRAINS_CFG.sub_terrains["hf_pyramid_slope_inv"].replace(
                    slope_range=(0.0, 0.18),
                    platform_width=0.40,
                    border_width=0.05,
                ),
            },
        ),
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    flat_terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    terrain_level_up_threshold = 0.8
    terrain_level_down_threshold = 0.3
    reset_height_offset = 0.08

    # joint names
    body_link_name = "Body"
    left_hip_joint_name = "Body_r_1"
    left_knee_joint_name = "L_leg1_r_4"
    left_wheel_joint_name = "L_leg2_r_7"
    right_hip_joint_name = "Body_r_8"
    right_knee_joint_name = "R_leg1_r_9"
    right_wheel_joint_name = "R_leg2_r_10"

    # default leg posture is now defined by branch hip angle a and mapped hip angle b
    default_branch_hip_angle = 0.5
    default_mapped_hip_angle = 0.3

    # observation noise
    enable_imu_noise = True
    imu_quat_noise_std = 0.01
    imu_ang_vel_noise_std = 0.08
    imu_projected_gravity_noise_std = 0.03

    # random disturbance push / torque pulses on body link
    enable_disturbance_force = False
    disturbance_force_scale = 1.5
    disturbance_force_probability = 0.02
    disturbance_force_duration_steps = 2
    enable_disturbance_torque = False
    disturbance_torque_scale = 0.015
    disturbance_torque_probability = 0.02
    disturbance_torque_duration_steps = 2

    # velocity command settings
    use_velocity_commands = True
    command_lin_vel_x_range = (-1.0, 1.0)
    command_yaw_vel_range = (-1.0, 1.0)
    command_resample_time_range = (1.0, 3.0)
    resample_commands = True
    command_lin_vel_x_min_abs = 0.15
    command_yaw_vel_min_abs = 0.35
    command_yaw_probability = 0.0
    command_tracking_sigma_lin = 0.08
    command_tracking_sigma_yaw = 0.2
    command_tracking_upright_sigma = 0.08
    command_tracking_stability_sigma = 0.5

    # action ranges
    mapped_hip_lower_limit = 0.0
    mapped_hip_upper_limit = 1.3962634015954636
    wheel_radius = 0.0325

    # non-wheel contact termination
    body_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/(Body|L_leg1|L_leg2|R_leg1|R_leg2)",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )
    termination_contact_force_threshold = 1.0

    # reward scales
    rew_scale_alive = 0.2
    rew_scale_terminated = -15.0
    rew_scale_angle = -1.0
    rew_scale_ang_vel = -0.15
    rew_scale_yaw_ang_vel = -0.2
    rew_scale_vertical_vel = -0.05
    rew_scale_servo_joint_vel = 0.0
    rew_scale_action_rate = 0.0
    rew_scale_servo_pose = -0.05
    rew_scale_leg_symmetry = -0.05
    rew_scale_wheel_spin = 0.0
    rew_scale_wheel_speed_diff = -0.002
    rew_scale_track_lin_vel = 5.0
    rew_scale_track_yaw_vel = 0.0
    rew_scale_track_wheel_lin = 2.0
    rew_scale_track_wheel_yaw = 0.0

    # max_body_angle = 0.5
