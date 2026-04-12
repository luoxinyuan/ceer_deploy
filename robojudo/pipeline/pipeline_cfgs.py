from typing import Any

from pydantic import model_validator

from robojudo.config import Config
from robojudo.controller import CtrlCfg
from robojudo.environment import EnvCfg
from robojudo.policy import PolicyCfg
from robojudo.tools.debug_log import DebugCfg


class PipelineCfg(Config):
    pipeline_type: str  # name of the pipeline class
    # ===== Pipeline Config =====
    device: str = "cpu"

    debug: DebugCfg = DebugCfg()

    run_fullspeed: bool = False
    """If True, run the pipeline at full speed, ignoring the desired frequency"""

    do_safety_check: bool = False
    """
    If True, perform safety check after each step.
    We recommend enabling this, however if motion is very aggressive, you may disable it.
    """


class RlPipelineCfg(PipelineCfg):
    pipeline_type: str = "RlPipeline"

    # ===== Pipeline Config =====
    robot: str  # robot name, e.g. "g1"

    env: EnvCfg | Any
    ctrl: list[CtrlCfg | Any] = []
    policy: PolicyCfg | Any
