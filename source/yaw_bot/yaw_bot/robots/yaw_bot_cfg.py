from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UsdFileCfg

YAW_BOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=UsdFileCfg(
        usd_path=r"D:\yaw\yaw_bot\assets\robots\yaw_bot\yaw_bot.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.18),
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
        # 4个舵机：位置控制
        "hip_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "Body_r_1",
                "Body_r_8",
            ],
            stiffness=18.0,
            damping=1.2,
            effort_limit=0.45,
            velocity_limit=8.5,
        ),
        "knee_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_leg1_r_4",
                "R_leg1_r_9",
            ],
            stiffness=24.0,
            damping=1.6,
            effort_limit=0.9,
            velocity_limit=8.5,
        ),
        # 2个轮子：力矩控制
        "wheel_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_leg2_r_7",
                "R_leg2_r_10",
            ],
            stiffness=0.0,
            damping=0.05,
            effort_limit=0.20,
            velocity_limit=125.0,
        ),
    },
)
