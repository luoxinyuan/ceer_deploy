import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco
import mujoco_viewer
import numpy as np

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.utils.util_func import quat_rotate_inverse_np, quatToEuler

logger = logging.getLogger(__name__)


@env_registry.register
class MujocoEnv(Environment):
    cfg_env: MujocoEnvCfg

    def __init__(self, cfg_env: MujocoEnvCfg, device="cpu"):
        super().__init__(cfg_env=cfg_env, device=device)

        self.sim_duration = cfg_env.sim_duration
        self.sim_dt = cfg_env.sim_dt
        self.sim_decimation = cfg_env.sim_decimation
        self.control_dt = self.sim_dt * self.sim_decimation

        logger.info("Loading Mujoco XML: %s", cfg_env.xml)
        self.model = mujoco.MjModel.from_xml_path(cfg_env.xml)  # pyright: ignore[reportAttributeAccessIssue]
        self._dof_qpos_indices, self._dof_qvel_indices = self._resolve_dof_indices()
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        # mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        self.viewer = mujoco_viewer.MujocoViewer(
            self.model,
            self.data,
            width=1200,
            height=900,
            hide_menus=True,
            diable_key_callbacks=True,
        )
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -10.0
        self.viewer.cam.azimuth = 180.0
        # self.viewer._paused = True

        if cfg_env.visualize_extras:
            self.visualizer = MujocoVisualizer(self.viewer)
        else:
            self.visualizer = None

        self._init_camera_capture()

        self.last_time = time.time()

        self.update()  # get initial state

    def reborn(self, init_qpos=None):
        if init_qpos is not None:
            self.data.qpos[0:7] = init_qpos
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
        else:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

    def reset(self):
        if self.born_place_align:  # TODO: merge
            self.born_place_align = False  # disable during reset
            self.update()
            self.born_place_align = True  # enable after reset
            self.set_born_place()
            self.update()

    def set_gains(self, stiffness, damping):
        assert len(stiffness) == self.num_dofs and len(damping) == self.num_dofs
        self.stiffness = np.asarray(stiffness)
        self.damping = np.asarray(damping)

    def self_check(self):
        pass

    def _resolve_dof_indices(self):
        qpos_indices = []
        qvel_indices = []
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                raise ValueError(f"Joint {joint_name} not found in Mujoco model.")
            qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))
            qvel_indices.append(int(self.model.jnt_dofadr[joint_id]))

        return np.asarray(qpos_indices, dtype=np.int32), np.asarray(qvel_indices, dtype=np.int32)

    def _init_camera_capture(self):
        self.camera_capture_enabled = bool(self.cfg_env.camera_capture_enabled)
        self.camera_renderer = None
        self.camera_image_writer = None
        self.camera_frame_i = 0
        self.next_camera_capture_time = time.time()

        if not self.camera_capture_enabled:
            return

        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cfg_env.camera_name)
        if camera_id == -1:
            logger.warning("Camera capture disabled: camera '%s' not found.", self.cfg_env.camera_name)
            self.camera_capture_enabled = False
            return

        try:
            import imageio.v2 as imageio
        except ImportError:
            logger.warning("Camera capture disabled: install imageio to save PNG frames.")
            self.camera_capture_enabled = False
            return

        output_dir = Path(self.cfg_env.camera_output_dir)
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        self.camera_output_dir = output_dir
        self.camera_image_writer = imageio
        self.camera_renderer = mujoco.Renderer(
            self.model,
            height=int(self.cfg_env.camera_image_height),
            width=int(self.cfg_env.camera_image_width),
        )
        logger.info(
            "Saving camera '%s' every %.2fs to %s",
            self.cfg_env.camera_name,
            self.cfg_env.camera_capture_interval_s,
            output_dir,
        )

    def _write_image_atomic(self, path: Path, rgb: np.ndarray):
        tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
        self.camera_image_writer.imwrite(tmp_path, rgb)
        tmp_path.replace(path)

    def _maybe_capture_camera(self):
        if not self.camera_capture_enabled:
            return

        now = time.time()
        if now < self.next_camera_capture_time:
            return

        interval_s = max(float(self.cfg_env.camera_capture_interval_s), 1e-3)
        self.next_camera_capture_time = now + interval_s
        self.camera_frame_i += 1

        self.camera_renderer.update_scene(self.data, camera=self.cfg_env.camera_name)
        rgb = self.camera_renderer.render()

        stamp_ms = int(now * 1000)
        frame_path = self.camera_output_dir / f"{self.cfg_env.camera_name}_{self.camera_frame_i:06d}_{stamp_ms}.png"
        latest_path = self.camera_output_dir / "latest.png"
        self._write_image_atomic(frame_path, rgb)
        self._write_image_atomic(latest_path, rgb)

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

    def update(self, simple=False):  # TODO: clean sensors in xml
        """simple: only update dof pos & vel"""
        dof_pos = self.data.qpos[self._dof_qpos_indices].astype(np.float32)
        dof_vel = self.data.qvel[self._dof_qvel_indices].astype(np.float32)

        self._dof_pos = dof_pos.copy()
        self._dof_vel = dof_vel.copy()

        if simple:
            return

        quat = self.data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]]
        ang_vel = self.data.qvel.astype(np.float32)[3:6]
        base_pos = self.data.qpos.astype(np.float32)[:3]
        lin_vel = self.data.qvel.astype(np.float32)[0:3]

        if self.born_place_align:
            quat, base_pos = self.base_align.align_transform(quat, base_pos)

        lin_vel = quat_rotate_inverse_np(quat, lin_vel)
        rpy = quatToEuler(quat)

        self._base_rpy = rpy.copy()
        self._base_quat = quat.copy()
        self._base_ang_vel = ang_vel.copy()

        self._base_pos = base_pos.copy()
        self._base_lin_vel = lin_vel.copy()

        if self.update_with_fk:
            fk_info = self.fk()
            self._fk_info = fk_info.copy()
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_pos = fk_info[self._torso_name]["pos"]

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"

        # print(f'pd_target: {pd_target}')

        if hand_pose is not None:
            logger.info("Hand pose-->", hand_pose)

        self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
        if self.viewer.is_alive:
            mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
            self.viewer.render()

        for _ in range(self.sim_decimation):
            torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)

            self.data.ctrl = torque

            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
            self.update(simple=True)
        self.update(simple=False)
        self._maybe_capture_camera()

    def shutdown(self):
        if self.camera_renderer is not None and hasattr(self.camera_renderer, "close"):
            self.camera_renderer.close()
        self.viewer.close()


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg

    mujoco_env = MujocoEnv(cfg_env=G1MujocoEnvCfg())
    mujoco_env.viewer._paused = False

    while True:
        # mujoco_env.update()
        mujoco_env.step(np.zeros(mujoco_env.num_dofs))
        time.sleep(0.02)
