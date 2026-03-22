# Yaw Bot（简体中文）

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

