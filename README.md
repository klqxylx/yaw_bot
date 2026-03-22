# Yaw Bot

`yaw_bot` is an Isaac Lab project for training and evaluating a two-wheel balancing robot with leg joints.

The repository currently contains:

- A direct RL task registered as `Template-Yaw-Bot-Direct-v0`
- An RSL-RL training and playback pipeline
- A simplified robot model under `assets/robots/yaw_bot`
- A leg-angle mapping utility for the equivalent parallel-leg geometry

## Project Layout

- [source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env.py](/d:/yaw/yaw_bot/source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env.py)
  Direct RL environment implementation
- [source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env_cfg.py](/d:/yaw/yaw_bot/source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env_cfg.py)
  Environment configuration, rewards, observations, terrain switch, disturbance settings
- [source/yaw_bot/yaw_bot/robots/yaw_bot_cfg.py](/d:/yaw/yaw_bot/source/yaw_bot/yaw_bot/robots/yaw_bot_cfg.py)
  Robot articulation and actuator configuration
- [source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/agents/rsl_rl_ppo_cfg.py](/d:/yaw/yaw_bot/source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/agents/rsl_rl_ppo_cfg.py)
  PPO runner configuration
- [scripts/rsl_rl/train.py](/d:/yaw/yaw_bot/scripts/rsl_rl/train.py)
  Training entry point
- [scripts/rsl_rl/play.py](/d:/yaw/yaw_bot/scripts/rsl_rl/play.py)
  Checkpoint playback entry point
- [scripts/calc_knee_angle.py](/d:/yaw/yaw_bot/scripts/calc_knee_angle.py)
  Small CLI utility for computing knee angle `t` from `a` and `b`

## Requirements

- Isaac Lab installed and working
- A Python environment that can import `isaaclab`, `isaaclab_tasks`, `isaaclab_rl`, and `rsl-rl-lib`
- Windows or Linux

This project is designed to be installed outside the main Isaac Lab repository, then imported by Isaac Lab.

## Installation

1. Install Isaac Lab first.

2. From the repository root, install the package in editable mode:

```bash
python -m pip install -e source/yaw_bot
```

If your Isaac Lab environment is launched through the provided shell wrapper, use that instead of plain `python`.

On Windows, a common pattern is:

```powershell
.\isaaclab.bat -p -m pip install -e source\yaw_bot
```

## Registered Task

The project currently registers one gym task:

```text
Template-Yaw-Bot-Direct-v0
```

You can verify registration with:

```powershell
python .\scripts\list_envs.py
```

## Training

Train with:

```powershell
python .\scripts\rsl_rl\train.py --task Template-Yaw-Bot-Direct-v0
```

Typical Windows Isaac Lab launch form:

```powershell
.\isaaclab.bat -p .\scripts\rsl_rl\train.py --task Template-Yaw-Bot-Direct-v0
```

Useful options:

- `--num_envs <N>`
- `--max_iterations <N>`
- `--seed <N>`
- `--video`

The current PPO experiment name is `yaw_bot_direct`, so new training logs are written under:

```text
logs/rsl_rl/yaw_bot_direct/<timestamp>/
```

Each run directory typically contains:

- `model_*.pt`
- `events.out.tfevents.*`
- `params/env.yaml`
- `params/agent.yaml`
- `exported/` after playback export

## Resume Training

Resume from a previous checkpoint with:

```powershell
python .\scripts\rsl_rl\train.py --task Template-Yaw-Bot-Direct-v0 --resume --load_run <run_name> --checkpoint <model_file>
```

Example:

```powershell
python .\scripts\rsl_rl\train.py --task Template-Yaw-Bot-Direct-v0 --resume --load_run 2026-03-18_13-26-33 --checkpoint model_999.pt
```

If the run was produced before the experiment name was renamed from the template default, its path may still be under:

```text
logs/rsl_rl/cartpole_direct/
```

## Play a Trained Policy

Run a saved checkpoint with:

```powershell
python .\scripts\rsl_rl\play.py --task Template-Yaw-Bot-Direct-v0 --checkpoint <absolute_or_relative_path_to_model>
```

Example:

```powershell
python .\scripts\rsl_rl\play.py --task Template-Yaw-Bot-Direct-v0 --checkpoint .\logs\rsl_rl\yaw_bot_direct\2026-03-18_17-19-18\model_700.pt
```

Current playback behavior:

- `play.py` forces a single environment
- termination is disabled during playback
- command resampling is disabled during playback
- the viewer is intended for manual keyboard control

Manual control keys in the simulation window:

- `W`
  Forward command
- `S`
  Backward command
- `A`
  Left yaw command
- `D`
  Right yaw command
- `L`
  Clear command to zero

The terminal also prints the current commanded linear and yaw speed whenever the manual command changes.

## Utility Script: Compute Knee Angle

The project includes a small helper for the current equivalent-leg mapping:

```powershell
python .\scripts\calc_knee_angle.py 10 20
```

This prints:

- branch hip angle `a` in degrees
- mapped hip angle `b` in degrees
- computed knee angle `t` in degrees

## Current Environment Notes

The current environment implementation includes:

- IMU-style observation noise
- Wheel encoder observations
- Last-action feedback
- Optional velocity-command observations for linear and yaw control
- Optional rough terrain through `use_rough_terrain`
- Optional random external body force and torque disturbances
- Non-wheel body contact termination through `ContactSensor`
- A control path where PPO outputs branch hip `a`, mapped hip `b`, and wheel torques
- Cylinder collision geometry for the wheels in the URDF

The current observation layout in command-tracking mode is:

- root quaternion: 4
- root angular velocity: 3
- projected gravity: 3
- velocity commands: 2
- wheel positions: 2
- wheel velocities: 2
- last actions: 6

Total: `22`

Some important config knobs live in:

- [yaw_bot_env_cfg.py](/d:/yaw/yaw_bot/source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env_cfg.py)
- [yaw_bot_cfg.py](/d:/yaw/yaw_bot/source/yaw_bot/yaw_bot/robots/yaw_bot_cfg.py)

## Actuators

The robot currently uses three actuator groups:

- `hip_joints`
  Left and right branch hip joints
- `knee_joints`
  Left and right simplified knee joints
- `wheel_joints`
  Left and right wheels

At the moment:

- Legs are controlled with `set_joint_position_target(...)`
- Wheels are controlled with `set_joint_effort_target(...)`

Current wheel collision is defined in the URDF with a cylinder collider using:

- diameter `65 mm`
- width `28 mm`

## Known Limitations

- The task id still uses the template-style prefix `Template-`
- Some older logs were written under the template experiment name `cartpole_direct`
- The project still contains template/example files such as [ui_extension_example.py](/d:/yaw/yaw_bot/source/yaw_bot/yaw_bot/ui_extension_example.py)
- The equivalent knee torque mapping function is present but not yet wired into leg control
- Older checkpoints may not resume if the observation dimension changes between runs
- The current command-training workflow is still staged: this repository often trains stand / forward / yaw in separate phases rather than all at once

## Training Diagnostics

Recent versions of the environment log wheel- and command-related diagnostics into TensorBoard. Useful tags include:

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

These are helpful when checking whether the robot is:

- actually rolling the wheels
- matching forward and backward commands with the correct sign
- slipping at the wheel-ground contact
- being over-constrained by the leg posture controller

## Quick Validation

Lightweight checks that are safe to run from the repository root:

```powershell
python .\scripts\calc_knee_angle.py 10 20
python -m py_compile .\source\yaw_bot\yaw_bot\tasks\direct\yaw_bot\yaw_bot_env.py
python -m py_compile .\source\yaw_bot\yaw_bot\tasks\direct\yaw_bot\yaw_bot_env_cfg.py
python -m py_compile .\source\yaw_bot\yaw_bot\robots\yaw_bot_cfg.py
```

## Development Notes

- Root-level [setup.py](/d:/yaw/yaw_bot/setup.py) exists, but the intended editable install path for the Isaac Lab extension package is still:

```text
source/yaw_bot
```

- The extension metadata lives in [extension.toml](/d:/yaw/yaw_bot/source/yaw_bot/config/extension.toml)

## License

This repository contains code derived from the Isaac Lab project template and keeps the original upstream headers in many files.
