# Yaw Bot（简体中文）

`yaw_bot` 是一个 Isaac Lab 项目，用于训练、评估和调试带腿关节的两轮平衡机器人。

Language: [English](./README.md)

`Yaw Bot` 这个名字有两层含义：

- `YAW` 表示偏航稳定与航向控制
- `YAW` 也可理解为 `You Always Walk`

项目当前重点包括：

- 平衡与站立
- 前进/后退速度指令跟踪
- 面向并联腿简化模型的等效腿角映射
- 轮地接触调试
- 基于 RSL-RL 的训练与回放

## 致谢 / 上游参考

本项目基于 StackForce 的开源双足轮式机器人方案：

- https://gitee.com/StackForce/bipedal_wheeled_robot

在本仓库中，我们额外使用了逆解映射函数，将并联腿结构转换为 Isaac Lab 中可训练、可控制的等效关节表示。

当前注册任务名：

```text
Template-Yaw-Bot-Direct-v0
```

## 仓库内容

- Isaac Lab Direct RL 任务实现
- yaw bot 机器人资产与关节/执行器配置
- RSL-RL 的 PPO 配置
- 训练与回放脚本
- 用于检查膝关节映射的小工具脚本

## 仓库结构

- [source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env.py](./source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env.py)  
  Direct RL 环境主实现
- [source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env_cfg.py](./source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env_cfg.py)  
  环境配置、指令、奖励、观测、地形与扰动开关
- [source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/agents/rsl_rl_ppo_cfg.py](./source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/agents/rsl_rl_ppo_cfg.py)  
  PPO Runner 配置
- [source/yaw_bot/yaw_bot/robots/yaw_bot_cfg.py](./source/yaw_bot/yaw_bot/robots/yaw_bot_cfg.py)  
  机器人关节与执行器定义
- [assets/robots/yaw_bot/yaw_bot.urdf](./assets/robots/yaw_bot/yaw_bot.urdf)  
  机器人 URDF（包含轮子碰撞几何）
- [assets/robots/yaw_bot/config.yaml](./assets/robots/yaw_bot/config.yaml)  
  资产转换配置
- [scripts/rsl_rl/train.py](./scripts/rsl_rl/train.py)  
  训练入口
- [scripts/rsl_rl/play.py](./scripts/rsl_rl/play.py)  
  回放入口
- [scripts/calc_knee_angle.py](./scripts/calc_knee_angle.py)  
  通过 `a` 与 `b` 计算膝关节角 `t` 的工具

## 依赖要求

- 已正确安装并可运行 Isaac Lab
- Python 环境可导入：
  - `isaaclab`
  - `isaaclab_tasks`
  - `isaaclab_rl`
  - `rsl_rl`
- Windows 或 Linux

本仓库设计为独立于 Isaac Lab 主仓库之外，以 editable 方式安装。

## 安装

在仓库根目录执行：

```powershell
python -m pip install -e source/yaw_bot
```

若你通过 Isaac Lab 启动器运行，可使用：

```powershell
.\isaaclab.bat -p -m pip install -e source\yaw_bot
```

## 验证任务注册

可通过下列命令查看已注册环境：

```powershell
python .\scripts\list_envs.py
```

检查输出中是否包含：

```text
Template-Yaw-Bot-Direct-v0
```

## 训练

基础训练命令：

```powershell
python .\scripts\rsl_rl\train.py --task Template-Yaw-Bot-Direct-v0
```

Windows 常见 Isaac Lab 启动形式：

```powershell
.\isaaclab.bat -p .\scripts\rsl_rl\train.py --task Template-Yaw-Bot-Direct-v0
```

常用参数：

- `--num_envs <N>`
- `--max_iterations <N>`
- `--seed <N>`
- `--video`

训练日志目录：

```text
logs/rsl_rl/yaw_bot_direct/<timestamp>/
```

通常包含：

- `model_*.pt`
- `events.out.tfevents.*`
- `params/env.yaml`
- `params/agent.yaml`
- `exported/`

## 继续训练（Resume）

从历史检查点继续训练：

```powershell
python .\scripts\rsl_rl\train.py --task Template-Yaw-Bot-Direct-v0 --resume --load_run <run_name> --checkpoint <model_file>
```

示例：

```powershell
python .\scripts\rsl_rl\train.py --task Template-Yaw-Bot-Direct-v0 --resume --load_run 2026-03-22_20-04-04 --checkpoint model_999.pt
```

说明：

- resume 会创建新的 run 目录，不会覆盖旧目录
- 若观测维度变化，旧检查点可能无法继续训练
- 某些历史日志仍可能位于：

```text
logs/rsl_rl/cartpole_direct/
```

## 策略回放

回放命令：

```powershell
python .\scripts\rsl_rl\play.py --task Template-Yaw-Bot-Direct-v0 --checkpoint <checkpoint_path>
```

示例：

```powershell
python .\scripts\rsl_rl\play.py --task Template-Yaw-Bot-Direct-v0 --checkpoint .\logs\rsl_rl\yaw_bot_direct\2026-03-22_20-04-04\model_999.pt
```

当前回放行为：

- 强制单环境
- 回放时禁用终止条件
- 回放时禁用随机重采样指令
- 设计为配合键盘手动控制

仿真窗口手动控制按键：

- `W`：前进指令
- `S`：后退指令
- `A`：左转偏航指令
- `D`：右转偏航指令
- `L`：清空指令

当手动指令变化时，终端会打印当前线速度/偏航速度指令。

## 工具脚本

辅助脚本：

- [calc_knee_angle.py](./scripts/calc_knee_angle.py)

用于由分支髋关节角 `a` 与映射髋关节角 `b` 计算膝关节角 `t`。

示例：

```powershell
python .\scripts\calc_knee_angle.py 10 20
```

输出包括：

- `a`（度）
- `b`（度）
- `t`（度）

## 机器人与控制概览

当前策略输出 6 维动作：

1. 左腿分支髋 `a`
2. 左腿映射髋 `b`
3. 右腿分支髋 `a`
4. 右腿映射髋 `b`
5. 左轮力矩指令
6. 右轮力矩指令

腿部控制：

- 策略输出 `a` 和 `b`
- 环境计算语义膝角 `t = f(a, b)`
- 仿真伺服目标为 `[left_hip, left_knee, right_hip, right_knee]`

轮子控制：

- 使用 `set_joint_effort_target(...)` 进行轮子力矩控制
- 环境统一了轮子符号约定，保证左右轮“语义前进”一致

## 并联腿等效映射

真实机构在仿真中按等效简化结构建模。

当前实现包含：

- 分支髋角 `a`
- 映射髋角 `b`
- 推导膝角 `t`

几何换算实现位于：

- [yaw_bot_env.py](./source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env.py)

膝关节力矩映射函数目前仅为占位接口，尚未完整接入腿部实际执行链路。

## 观测（Observations）

当前命令跟踪模式下观测维度为 `22`。

构成为：

- 根部四元数：4
- 根部角速度：3
- 投影重力：3
- 速度指令：2
- 轮子位置：2
- 轮子速度：2
- 上一步动作：6

总计：

```text
22
```

观测配置位于：

- [yaw_bot_env_cfg.py](./source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env_cfg.py)

## 指令（Commands）

环境通过下列开关启用命令跟踪模式：

- `use_velocity_commands`

当前指令集：

- x 方向线速度指令
- z 轴偏航角速度指令

训练时由环境采样；回放时由手动输入写入。

相关配置字段：

- `command_lin_vel_x_range`
- `command_yaw_vel_range`
- `command_resample_time_range`
- `command_lin_vel_x_min_abs`
- `command_yaw_vel_min_abs`
- `command_yaw_probability`

## 奖励（Rewards）

当前奖励结构包含：

- 存活奖励
- 终止惩罚
- 姿态角惩罚
- 角速度惩罚
- 垂向速度惩罚
- 可选的腿部姿态与对称性正则
- 线速度指令跟踪奖励
- 基于轮子语义速度的线速度跟踪奖励

主要奖励配置在：

- [yaw_bot_env_cfg.py](./source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env_cfg.py)

具体实现在：

- [yaw_bot_env.py](./source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env.py)

## 执行器（Actuators）

机器人使用三组执行器：

- `hip_joints`
- `knee_joints`
- `wheel_joints`

当前控制分工：

- 髋与膝使用位置目标
- 轮子使用力矩（effort）目标

执行器配置位于：

- [yaw_bot_cfg.py](./source/yaw_bot/yaw_bot/robots/yaw_bot_cfg.py)

## 接触与终止（Contact and Termination）

终止逻辑基于 `ContactSensor` 检测到的非轮子机体接触。

用于终止检测的链接包括：

- `Body`
- `L_leg1`
- `L_leg2`
- `R_leg1`
- `R_leg2`

轮子接触允许。

## 地形与扰动（Terrain and Disturbances）

环境支持以下可选项：

- 平地
- 粗糙地形
- IMU 噪声
- 随机机体力脉冲
- 随机机体力矩脉冲

上述开关均在以下配置中：

- [yaw_bot_env_cfg.py](./source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/yaw_bot_env_cfg.py)

典型训练流程建议分阶段：

1. 先在平地训练站立或直线
2. 再加入前后速度跟踪
3. 再加入偏航控制
4. 最后加入扰动和粗糙地形

## 轮子几何与接触

URDF 中轮子碰撞几何当前使用圆柱，而非网格碰撞体。

当前碰撞尺寸：

- 直径 `65 mm`
- 宽度 `28 mm`

定义文件：

- [yaw_bot.urdf](./assets/robots/yaw_bot/yaw_bot.urdf)

改为圆柱碰撞体的主要目的是改善相较于网格凸包的滚动接地行为。

## 训练诊断指标（Training Diagnostics）

环境新版本会将轮子与指令相关诊断写入 TensorBoard。

常用 tag 包括：

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

这些指标可用于判断：

- 轮子是否真的在驱动
- 前后指令是否学到正确方向
- 是否由轮地打滑主导
- 腿部控制是否过度约束推进

## 已知限制（Current Known Limitations）

- gym 任务名仍为模板前缀 `Template-`
- 一些旧日志仍在旧实验名 `cartpole_direct` 下
- 偏航控制通常后于直线控制训练，并非所有检查点都能稳定转向
- 若观测维度变化，旧检查点可能无法加载
- 等效膝力矩映射尚未接入实际腿部执行链路
- 仓库仍包含模板示例文件，例如 [ui_extension_example.py](./source/yaw_bot/yaw_bot/ui_extension_example.py)

## 快速校验（Quick Validation）

在仓库根目录可做轻量检查：

```powershell
python .\scripts\calc_knee_angle.py 10 20
python -m py_compile .\source\yaw_bot\yaw_bot\tasks\direct\yaw_bot\yaw_bot_env.py
python -m py_compile .\source\yaw_bot\yaw_bot\tasks\direct\yaw_bot\yaw_bot_env_cfg.py
python -m py_compile .\source\yaw_bot\yaw_bot\robots\yaw_bot_cfg.py
```

## 开发备注（Development Notes）

- 推荐 editable install 包路径：

```text
source/yaw_bot
```

- 仓库根目录虽然存在 [setup.py](./setup.py)，但实际 Isaac Lab 扩展包在：

```text
source/yaw_bot
```

- 扩展元数据位于：
  - [extension.toml](./source/yaw_bot/config/extension.toml)

## License

本仓库含有源自 Isaac Lab 模板的代码，许多文件仍保留上游版权头与许可证声明。
`yaw_bot` 是一个基于 Isaac Lab 的两轮双足平衡机器人项目，用于训练、评估与调试强化学习策略。

English version: [README.md](./README.md)

## 项目来源与声明

本项目使用了 StackForce 开源的双足轮式机器人方案作为基础参考：

- https://gitee.com/StackForce/bipedal_wheeled_robot

在此基础上，本仓库针对并联腿结构加入了**逆解函数（inverse-solution mapping）**，用于将并联机构等效为训练环境中更易控制的关节表示，并用于动作映射与奖励构造相关计算。

## 当前实现特点

- 基于 Isaac Lab 的 Direct RL 任务实现
- 面向 RSL-RL 的训练与回放脚本
- 支持前进/后退与偏航速度指令跟踪
- 包含并联腿结构的等效角度映射（含膝关节等效计算）
- 提供轮地接触与姿态稳定相关调试能力

## 快速开始

1. 安装扩展：

```bash
python -m pip install -e source/yaw_bot
```

2. 训练：

```bash
python ./scripts/rsl_rl/train.py --task Template-Yaw-Bot-Direct-v0
```

3. 回放：

```bash
python ./scripts/rsl_rl/play.py --task Template-Yaw-Bot-Direct-v0
```

## 主要目录

- `source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/`：环境与任务配置
- `source/yaw_bot/yaw_bot/robots/`：机器人关节与执行器配置
- `scripts/rsl_rl/`：训练与回放入口脚本
- `assets/robots/yaw_bot/`：机器人资源与配置

