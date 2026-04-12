from pydantic import field_validator, model_validator

from robojudo.config import ASSETS_DIR, Config
from robojudo.tools.tool_cfgs import DoFConfig


class PolicyCfg(Config):
    policy_type: str
    robot: str

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/PLCAEHOLDER.pt"
        return policy_file.as_posix()

    disable_autoload: bool = False
    freq: int = 50

    obs_dof: DoFConfig
    action_dof: DoFConfig

    action_scale: float = 1.0
    action_clip: float | None = None
    action_beta: float = 1.0

    history_length: int = 0

    @property
    def history_obs_size(self) -> int:
        return 0

    @field_validator("action_scale", "action_clip")
    def check_action_scale(cls, v):
        if v is not None and v <= 0:
            raise ValueError("action_scale must be positive")
        return v

    @model_validator(mode="after")
    def check_history(self):
        if self.history_length < 0:
            raise ValueError("history_length cannot be negative")
        if self.history_obs_size < 0:
            raise ValueError("history_obs_size cannot be negative")
        return self
