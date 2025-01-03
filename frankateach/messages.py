from dataclasses import dataclass
import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation as R


@dataclass
class FrankaState:
    pos: np.ndarray
    quat: np.ndarray
    gripper: np.ndarray
    timestamp: float
    start_teleop: bool = False


@dataclass
class FrankaAction:
    pos: np.ndarray
    quat: np.ndarray
    gripper: np.ndarray
    reset: bool
    timestamp: float


@dataclass
class ControllerState:
    created_timestamp: float

    left_x: bool
    left_y: bool
    left_menu: bool
    left_thumbstick: bool
    left_index_trigger: float
    left_hand_trigger: float
    left_thumbstick_axes: np.ndarray[Tuple[float, float]]
    left_local_position: np.ndarray[Tuple[float, float, float]]
    left_local_rotation: np.ndarray[Tuple[float, float, float, float]]

    right_a: bool
    right_b: bool
    right_menu: bool
    right_thumbstick: bool
    right_index_trigger: float
    right_hand_trigger: float
    right_thumbstick_axes: np.ndarray[Tuple[float, float]]
    right_local_position: np.ndarray[Tuple[float, float, float]]
    right_local_rotation: np.ndarray[Tuple[float, float, float, float]]

    @property
    def right_position(self) -> np.ndarray:
        return self.right_affine[:3, 3]

    @property
    def left_position(self) -> np.ndarray:
        return self.left_affine[:3, 3]

    @property
    def right_rotation_matrix(self) -> np.ndarray:
        return self.right_affine[:3, :3]

    @property
    def left_rotation_matrix(self) -> np.ndarray:
        return self.left_affine[:3, :3]

    @property
    def left_affine(self) -> np.ndarray:
        return self.get_affine(self.left_local_position, self.left_local_rotation)

    @property
    def right_affine(self) -> np.ndarray:
        return self.get_affine(self.right_local_position, self.right_local_rotation)

    def get_affine(
        self, controller_position: np.ndarray, controller_rotation: np.ndarray
    ):
        """Returns a 4x4 affine matrix from the controller's position and rotation.
        Args:
            controller_position: 3D position of the controller.
            controller_rotation: 4D quaternion of the controller's rotation.

            All in headset space.
        """

        return np.block(
            [
                [
                    R.as_matrix(R.from_quat(controller_rotation)),
                    controller_position[:, np.newaxis],
                ],
                [np.zeros((1, 3)), 1.0],
            ]
        )
