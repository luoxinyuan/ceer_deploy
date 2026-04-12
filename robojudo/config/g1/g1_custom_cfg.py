from robojudo.config import cfg_registry
from robojudo.config.g1.policy.g1_my_custom_policy_cfg import G1MyCustomPolicyCfg
from robojudo.controller.ctrl_cfgs import (
    JoystickCtrlCfg,
    KeyboardCtrlCfg,
)
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg

from .env.g1_mujuco_env_cfg import G1MujocoEnvCfg


@cfg_registry.register
class g1_my_rl(RlPipelineCfg):
    """
    Minimal sim2sim configuration for the custom G1 policy.
    """
    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg] = [
        JoystickCtrlCfg(),
        KeyboardCtrlCfg(),
    ]
    policy: G1MyCustomPolicyCfg = G1MyCustomPolicyCfg()
