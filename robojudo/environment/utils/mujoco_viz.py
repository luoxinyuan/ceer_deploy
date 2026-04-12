import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as sRot


class MujocoVisualizer:
    def __init__(self, viewer):
        self.viewer = viewer
        self.model = viewer.model
        self.data = viewer.data
        self._mocap_id_cache = {}
        self._geom_id_cache = {}

    # TODO: reset: clear all markers

    def _get_mocap_id(self, body_name: str) -> int:
        if body_name not in self._mocap_id_cache:
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,  # pyright: ignore[reportAttributeAccessIssue]
                body_name,
            )
            if body_id < 0:
                raise KeyError(f"Body not found for visualization: {body_name}")
            mocap_id = int(self.model.body_mocapid[body_id])
            if mocap_id < 0:
                raise KeyError(f"Body is not mocap-enabled: {body_name}")
            self._mocap_id_cache[body_name] = mocap_id
        return self._mocap_id_cache[body_name]

    def _get_geom_id(self, geom_name: str) -> int:
        if geom_name not in self._geom_id_cache:
            geom_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,  # pyright: ignore[reportAttributeAccessIssue]
                geom_name,
            )
            if geom_id < 0:
                raise KeyError(f"Geom not found for visualization: {geom_name}")
            self._geom_id_cache[geom_name] = geom_id
        return self._geom_id_cache[geom_name]

    def set_mocap_pose(self, body_name, pos, quat_xyzw=None):
        mocap_id = self._get_mocap_id(body_name)
        self.data.mocap_pos[mocap_id] = np.asarray(pos, dtype=float)
        if quat_xyzw is not None:
            quat_xyzw = np.asarray(quat_xyzw, dtype=float)
            quat_wxyz = np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
                dtype=float,
            )
            self.data.mocap_quat[mocap_id] = quat_wxyz

    def hide_mocap(self, body_name, hidden_pos=None):
        if hidden_pos is None:
            hidden_pos = np.array([0.0, 0.0, -10.0], dtype=float)
        self.set_mocap_pose(body_name, hidden_pos)

    def set_arrow_length(self, shaft_geom_name, tip_geom_name, length, shaft_radius=0.012):
        geom_len = max(float(length), 0.05)
        shaft_id = self._get_geom_id(shaft_geom_name)
        tip_id = self._get_geom_id(tip_geom_name)

        self.model.geom_size[shaft_id] = np.array([0.5 * geom_len, shaft_radius, shaft_radius], dtype=float)
        self.model.geom_pos[shaft_id] = np.array([0.5 * geom_len, 0.0, 0.0], dtype=float)
        self.model.geom_pos[tip_id] = np.array([geom_len, 0.0, 0.0], dtype=float)

    def draw_sphere(self, pos, radius, color, id=0):
        self.viewer.add_marker(
            pos=np.asarray(pos, dtype=float),
            size=np.array([radius, radius, radius], dtype=float),
            rgba=np.asarray(color, dtype=float),
            type=mujoco.mjtGeom.mjGEOM_SPHERE,  # pyright: ignore[reportAttributeAccessIssue]
            label="",
            id=1000 + id,
        )

    def update_rg_view(self, body_pos, body_rot, humanoid_id):
        if humanoid_id == 0:
            rgba = (1, 0, 0, 1)
        elif humanoid_id == 1:
            rgba = (0, 1, 0, 1)
        else:
            return

        for j in range(body_pos.shape[0]):
            # need to modify mujoco_viewer to support this
            self.viewer.add_marker(
                pos=body_pos[j],
                size=0.05,
                rgba=rgba,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,  # pyright: ignore[reportAttributeAccessIssue]
                label="",
                id=humanoid_id * 1000 + j,
            )

    def draw_arrow(self, origin, root_quat, vec_local, color, scale=1.0, horizontal_only=False, id=0):
        vec_local = np.array(vec_local, dtype=float)
        r = sRot.from_quat(root_quat)

        if horizontal_only:
            yaw = r.as_euler("xyz", degrees=False)[2]
            yaw_rot = sRot.from_euler("z", yaw).as_matrix()
            vec_world = yaw_rot @ vec_local
        else:
            vec_world = r.as_matrix() @ vec_local

        length = np.linalg.norm(vec_world)
        if length > 1e-6:
            dir_world = vec_world / length
            scaled_length = length * scale

            rot, _ = sRot.align_vectors([dir_world], [[0, 0, 1]])
            mat = rot.as_matrix()

            center_pos = origin  # dir_world * scaled_length * 0.5
        else:
            # zero length
            scaled_length = 0
            mat = np.eye(3)
            center_pos = origin

        self.viewer.add_marker(
            pos=center_pos,
            mat=mat,
            size=np.array([0.02, 0.02, scaled_length]),
            rgba=np.array(color),
            type=mujoco.mjtGeom.mjGEOM_ARROW,  # pyright: ignore[reportAttributeAccessIssue]
            id=3000 + id,
        )
