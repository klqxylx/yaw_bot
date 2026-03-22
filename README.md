# Yaw Bot

`yaw_bot` is an Isaac Lab project for training, evaluating, and debugging a two-wheel balancing robot with leg joints.

Language: [简体中文](./README_cn.md)

The name `Yaw Bot` carries two meanings:

- `YAW` refers to yaw-angle stability and heading control
- `YAW` also stands for `You Always Walk`

The project currently focuses on:

- balancing and standing
- forward/backward command tracking
- equivalent leg-angle mapping for a simplified parallel-leg model
- wheel-ground contact debugging
- RSL-RL based training and playback

## Acknowledgement / Upstream Reference

This project is based on the open-source bipedal wheeled robot from StackForce:

- https://gitee.com/StackForce/bipedal_wheeled_robot

In this repository, we additionally use an inverse-solution mapping function to convert the parallel-leg structure into an equivalent joint representation for training and control in Isaac Lab.

The registered task is:

```text
Template-Yaw-Bot-Direct-v0
```

## What This Repository Contains

- a direct RL task implementation for Isaac Lab
- a robot asset and articulation config for the yaw bot
- PPO configs for RSL-RL
- training and playback scripts
- a small utility script for checking the current knee-angle mapping

## Repository Layout

- [source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env.py](./source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env.py)
  Main direct RL environment
- [source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env_cfg.py](./source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env_cfg.py)
  Environment configuration, commands, rewards, observations, terrain and disturbance switches
- [source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/agents/rsl_rl_ppo_cfg.py](./source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/agents/rsl_rl_ppo_cfg.py)
  PPO runner configuration
- [source/yaw_bot/yaw_bot/robots/yaw_bot_cfg.py](./source/yaw_bot/yaw_bot/robots/yaw_bot_cfg.py)
  Robot articulation and actuator definitions
- [assets/robots/yaw_bot/yaw_bot.urdf](./assets/robots/yaw_bot/yaw_bot.urdf)
  Robot URDF, including wheel collision geometry
- [assets/robots/yaw_bot/config.yaml](./assets/robots/yaw_bot/config.yaml)
  Asset conversion configuration
- [scripts/rsl_rl/train.py](./scripts/rsl_rl/train.py)
  Training entry point
- [scripts/rsl_rl/play.py](./scripts/rsl_rl/play.py)
  Playback entry point
- [scripts/calc_knee_angle.py](./scripts/calc_knee_angle.py)
  Helper for computing knee angle `t` from `a` and `b`

## Requirements

- Isaac Lab installed and working
- a Python environment that can import:
  - `isaaclab`
  - `isaaclab_tasks`
  - `isaaclab_rl`
  - `rsl_rl`
- Windows or Linux

This repository is intended to live outside the main Isaac Lab repository and be installed as an editable package.

## Installation

Install the extension package from the repository root:

```powershell
python -m pip install -e source/yaw_bot
```

If you normally launch inside the Isaac Lab wrapper, use:

```powershell
.\isaaclab.bat -p -m pip install -e source\yaw_bot
```

## Verify Task Registration

You can list registered environments with:

```powershell
python .\scripts\list_envs.py
```

Look for:

```text
Template-Yaw-Bot-Direct-v0
```

## Training

Basic training command:

```powershell
python .\scripts\rsl_rl\train.py --task Template-Yaw-Bot-Direct-v0
```

Typical Isaac Lab launch form on Windows:

```powershell
.\isaaclab.bat -p .\scripts\rsl_rl\train.py --task Template-Yaw-Bot-Direct-v0
```

Useful overrides:

- `--num_envs <N>`
- `--max_iterations <N>`
- `--seed <N>`
- `--video`

Training logs are written under:

```text
logs/rsl_rl/yaw_bot_direct/<timestamp>/
```

Each run directory typically contains:

- `model_*.pt`
- `events.out.tfevents.*`
- `params/env.yaml`
- `params/agent.yaml`
- `exported/`

## Resume Training

Resume from a previous checkpoint with:

```powershell
python .\scripts\rsl_rl\train.py --task Template-Yaw-Bot-Direct-v0 --resume --load_run <run_name> --checkpoint <model_file>
```

Example:

```powershell
python .\scripts\rsl_rl\train.py --task Template-Yaw-Bot-Direct-v0 --resume --load_run 2026-03-22_20-04-04 --checkpoint model_999.pt
```

Notes:

- resuming creates a new run directory rather than overwriting the old one
- old checkpoints may fail to resume if the observation dimension changed
- some older runs may still live under:

```text
logs/rsl_rl/cartpole_direct/
```

## Play a Trained Policy

Playback command:

```powershell
python .\scripts\rsl_rl\play.py --task Template-Yaw-Bot-Direct-v0 --checkpoint <checkpoint_path>
```

Example:

```powershell
python .\scripts\rsl_rl\play.py --task Template-Yaw-Bot-Direct-v0 --checkpoint .\logs\rsl_rl\yaw_bot_direct\2026-03-22_20-04-04\model_999.pt
```

Current playback behavior:

- playback forces a single environment
- termination is disabled during playback
- command resampling is disabled during playback
- playback is designed for manual keyboard command input

Manual control keys in the simulation window:

- `W`
  forward command
- `S`
  backward command
- `A`
  left yaw command
- `D`
  right yaw command
- `L`
  clear command

The terminal prints the current commanded linear and yaw speed whenever the manual command changes.

## Utility Script

The helper script:

- [calc_knee_angle.py](./scripts/calc_knee_angle.py)

computes the knee angle `t` from branch hip angle `a` and mapped hip angle `b`.

Example:

```powershell
python .\scripts\calc_knee_angle.py 10 20
```

This prints:

- `a` in degrees
- `b` in degrees
- `t` in degrees

## Robot and Control Overview

The current policy outputs 6 actions:

1. left branch hip `a`
2. left mapped hip `b`
3. right branch hip `a`
4. right mapped hip `b`
5. left wheel torque command
6. right wheel torque command

Leg control:

- the policy outputs `a` and `b`
- the environment computes the semantic knee angle `t = f(a, b)`
- the simulated servo targets become `[left_hip, left_knee, right_hip, right_knee]`

Wheel control:

- wheels are controlled with `set_joint_effort_target(...)`
- wheel sign conventions are unified in the environment so semantic forward wheel motion is consistent across left and right wheels

## Equivalent Leg Mapping

The real mechanism is treated as an equivalent simplified structure in simulation.

Current implementation includes:

- branch hip angle `a`
- mapped hip angle `b`
- derived knee angle `t`

The geometry conversion is implemented in:

- [yaw_bot_env.py](./source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env.py)

The knee torque mapping function exists as a placeholder interface, but torque-equivalent actuation is not yet fully wired into the leg control path.

## Observations

Current command-tracking observation size is `22`.

The layout is:

- root quaternion: 4
- root angular velocity: 3
- projected gravity: 3
- velocity commands: 2
- wheel positions: 2
- wheel velocities: 2
- last actions: 6

Total:

```text
22
```

The observation configuration lives in:

- [yaw_bot_env_cfg.py](./source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env_cfg.py)

## Commands

The environment supports command-tracking mode through:

- `use_velocity_commands`

The current command set is:

- linear x velocity command
- yaw angular velocity command

These commands are sampled inside the environment during training and written manually during playback.

Relevant config fields:

- `command_lin_vel_x_range`
- `command_yaw_vel_range`
- `command_resample_time_range`
- `command_lin_vel_x_min_abs`
- `command_yaw_vel_min_abs`
- `command_yaw_probability`

## Rewards

The current reward structure includes:

- alive reward
- termination penalty
- body angle penalty
- angular-velocity penalties
- vertical-velocity penalty
- optional leg pose and symmetry regularization
- linear command-tracking reward
- wheel-based linear command-tracking reward

The main reward config is in:

- [yaw_bot_env_cfg.py](./source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env_cfg.py)

The actual implementation is in:

- [yaw_bot_env.py](./source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env.py)

## Actuators

The robot uses three actuator groups:

- `hip_joints`
- `knee_joints`
- `wheel_joints`

Current control split:

- hips and knees use position targets
- wheels use effort targets

Current actuator config is in:

- [yaw_bot_cfg.py](./source/yaw_bot/yaw_bot/robots/yaw_bot_cfg.py)

## Contact and Termination

Termination is based on non-wheel body contact through a `ContactSensor`.

Tracked links for termination include:

- `Body`
- `L_leg1`
- `L_leg2`
- `R_leg1`
- `R_leg2`

Wheel contact is allowed.

## Terrain and Disturbances

The environment supports these optional features:

- flat terrain
- rough terrain
- IMU noise
- random body force pulses
- random body torque pulses

These are all controlled in:

- [yaw_bot_env_cfg.py](./source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env_cfg.py)

Typical workflow is staged:

1. train standing or straight-line motion on flat terrain
2. add forward/backward command tracking
3. add yaw control
4. add disturbances and rough terrain

## Wheel Geometry and Contact

The wheel collision geometry in the URDF is currently a cylinder, not a mesh collider.

Current wheel collision dimensions:

- diameter `65 mm`
- width `28 mm`

This is defined in:

- [yaw_bot.urdf](./assets/robots/yaw_bot/yaw_bot.urdf)

This change was made to improve wheel-ground rolling behavior compared with mesh-based convex hull collision.

## Training Diagnostics

Recent versions of the environment log wheel- and command-related diagnostics to TensorBoard.

Useful tags include:

- `Diagnostics/root_lin_vel_x`
- `Diagnostics/wheel_semantic_forward_vel`
- `Diagnostics/wheel_effort_cmd_abs`
- `Diagnostics/wheel_surface_speed`
- `Diagnostics/wheel_body_speed_slip_abs`
- `Diagnostics/lin_cmd_sign_match_rate`
- `Diagnostics/forward_cmd_success_rate`
- `Diagnostics/backward_cmd_success_rate`
- `Diagnostics/servo_pose_error`
- `Diagnostics/servo_joint_vel_sq`
- `Diagnostics/gravity_xy_error`
- `Diagnostics/root_vertical_vel_abs`

These are especially useful for checking:

- whether the wheels are actually being driven
- whether forward and backward commands are learned with the correct sign
- whether wheel-ground slip is dominating
- whether the leg controller is over-constraining propulsion

## Current Known Limitations

- the gym task name still uses the template-style prefix `Template-`
- some older logs remain under the old template experiment name `cartpole_direct`
- yaw control is often trained later than forward/backward control, so not every checkpoint can turn
- old checkpoints may not load if observation dimensions changed
- the equivalent knee torque mapping is not yet part of the actual leg actuation path
- the repository still contains some template/example files, such as [ui_extension_example.py](./source/yaw_bot/yaw_bot/ui_extension_example.py)

## Quick Validation

Useful lightweight checks from the repository root:

```powershell
python .\scripts\calc_knee_angle.py 10 20
python -m py_compile .\source\yaw_bot\yaw_bot\tasks\direct\yaw_bot\yaw_bot_env.py
python -m py_compile .\source\yaw_bot\yaw_bot\tasks\direct\yaw_bot\yaw_bot_env_cfg.py
python -m py_compile .\source\yaw_bot\yaw_bot\robots\yaw_bot_cfg.py
```

## Development Notes

- the intended editable install package is:

```text
source/yaw_bot
```

- root-level [setup.py](./setup.py) exists, but the actual Isaac Lab extension package is under:

```text
source/yaw_bot
```

- extension metadata lives in:
  - [extension.toml](./source/yaw_bot/config/extension.toml)

## License

This repository contains code derived from the Isaac Lab project template and retains the original upstream headers in many files.
