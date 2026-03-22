import argparse

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# launch app
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Debug yaw_bot axis directions.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# imports
# -----------------------------------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

# -----------------------------------------------------------------------------
# robot cfg
# -----------------------------------------------------------------------------
YAW_BOT_CFG = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=r"D:\yaw\yaw_bot\assets\robots\yaw_bot\yaw_bot.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.12),
        joint_pos={
            "Body_r_1": 0.3,
            "L_leg1_r_4": 0.8,
            "L_leg2_r_7": 0.0,
            "Body_r_8": 0.3,
            "R_leg1_r_9": -0.8,
            "R_leg2_r_10": 0.0,
        },
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
            effort_limit=1.0,
            velocity_limit=200.0,
        ),
    },
)


def main():
    sim_cfg = SimulationCfg(dt=1 / 120)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([1.5, 1.5, 1.0], [0.0, 0.0, 0.2])

    spawn_ground_plane("/World/ground", GroundPlaneCfg())

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(YAW_BOT_CFG)

    sim.reset()

    print("\n====== DEBUG AXIS ======")
    print("root_ang_vel_w = [wx, wy, wz]")
    print("wx = rotation about X axis")
    print("wy = rotation about Y axis")
    print("wz = rotation about Z axis")
    print("========================\n")

    print("[INFO] Joint names:")
    for i, name in enumerate(robot.joint_names):
        print(f"{i}: {name}")
    print()

    step = 0
    while simulation_app.is_running():
        sim.step()

        if step % 120 == 0:
            ang_vel = robot.data.root_ang_vel_w[0].cpu().tolist()
            lin_vel = robot.data.root_lin_vel_w[0].cpu().tolist()

            print(f"step: {step}")
            print("root_ang_vel_w:", ang_vel)
            print("root_lin_vel_w:", lin_vel)
            print()

        step += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
