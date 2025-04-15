import time
import pickle
from frankateach.utils import notify_component_start
from frankateach.network import (
    ZMQKeypointSubscriber,
    create_request_socket,
    ZMQKeypointPublisher,
)
from frankateach.constants import (
    COMMANDED_STATE_PORT,
    CONTROL_PORT,
    HOST,
    STATE_PORT,
    VR_CONTROLLER_STATE_PORT,
    H_R_V,
    H_R_V_star,
    ROBOT_WORKSPACE_MIN,
    ROBOT_WORKSPACE_MAX,
    GRIPPER_OPEN,
    GRIPPER_CLOSE,
)
from frankateach.messages import FrankaAction, FrankaState

from deoxys.utils import transform_utils

import numpy as np
from numpy.linalg import pinv


def get_relative_affine(init_affine, current_affine, deoxys_config_path):

    H_V_des = pinv(init_affine) @ current_affine

    # Transform to robot frame.
    relative_affine_rot = (pinv(H_R_V) @ H_V_des @ H_R_V)[:3, :3]
    relative_affine_trans = (pinv(H_R_V_star) @ H_V_des @ H_R_V_star)[:3, 3]

    # Mirror the x-axis and y-axis for left arm setup
    if "deoxys_left" in deoxys_config_path:
        relative_affine_trans[0] *= -1  
        relative_affine_trans[1] *= -1 
        relative_affine_rot[:, 0] *= -1  
        relative_affine_rot[:, 1] *= -1
        
    # Homogeneous coordinates
    relative_affine = np.block(
        [[relative_affine_rot, relative_affine_trans.reshape(3, 1)], [0, 0, 0, 1]]
    )

    return relative_affine


class FrankaOperator:
    def __init__(
        self,
        init_gripper_state="open",
        teleop_mode="robot",
        home_offset=[0, 0, 0],
        deoxys_config_path=None,
        continuous_gripper=False,   
    ) -> None:
        # Subscribe controller state
        self._controller_state_subscriber = ZMQKeypointSubscriber(
            host=HOST, port=VR_CONTROLLER_STATE_PORT, topic="controller_state"
        )

        self.action_socket = create_request_socket(HOST, CONTROL_PORT)
        self.state_socket = ZMQKeypointPublisher(HOST, STATE_PORT)
        self.commanded_state_socket = ZMQKeypointPublisher(HOST, COMMANDED_STATE_PORT)

        # Class variables
        # self._save_states = save_states
        self.is_first_frame = True
        self.gripper_state = (
            GRIPPER_OPEN if init_gripper_state == "open" else GRIPPER_CLOSE
        )
        self.start_teleop = False
        self.init_affine = None
        self.teleop_mode = teleop_mode

        if teleop_mode == "human" and home_offset is None:
            home_offset = [-0.22, 0.0, 0.1]
        self.home_offset = (
            np.array(home_offset) if home_offset is not None else np.zeros(3)
        )
        self.deoxys_config_path = deoxys_config_path
        self.continuous_gripper = continuous_gripper
    def _apply_retargeted_angles(self) -> None:
        self.controller_state = self._controller_state_subscriber.recv_keypoints()

        if self.is_first_frame:
            print("Resetting robot..")
            action = FrankaAction(
                pos=np.zeros(3),
                quat=np.zeros(4),
                gripper=self.gripper_state,
                reset=True,
                timestamp=time.time(),
            )
            self.action_socket.send(bytes(pickle.dumps(action, protocol=-1)))
            robot_state = pickle.loads(self.action_socket.recv())

            # Move to offset position
            target_pos = robot_state.pos + self.home_offset
            target_quat = robot_state.quat
            action = FrankaAction(
                pos=target_pos.flatten().astype(np.float32),
                quat=target_quat.flatten().astype(np.float32),
                gripper=self.gripper_state,
                reset=False,
                timestamp=time.time(),
            )
            self.action_socket.send(bytes(pickle.dumps(action, protocol=-1)))
            robot_state = pickle.loads(self.action_socket.recv())
            # HOME <- Pos: [0.457632  0.0321814 0.2653815], Quat: [0.9998586  0.00880853 0.01421072 0.00179784]

            print(robot_state)
            self.home_rot, self.home_pos = (
                transform_utils.quat2mat(robot_state.quat),
                robot_state.pos,
            )

            self.is_first_frame = False
        if self.controller_state.right_a:
            self.start_teleop = True
            self.init_affine = self.controller_state.right_affine
        if self.controller_state.right_b:
            self.start_teleop = False
            self.init_affine = None
            # receive the robot state
            self.action_socket.send(b"get_state")
            robot_state: FrankaState = pickle.loads(self.action_socket.recv())
            if robot_state == b"state_error":
                print("Error getting robot state")
                return

            self.home_rot, self.home_pos = (
                transform_utils.quat2mat(robot_state.quat),
                robot_state.pos,
            )

        if self.start_teleop and self.teleop_mode == "robot":
            relative_affine = get_relative_affine(
                self.init_affine, self.controller_state.right_affine, self.deoxys_config_path
            )
        else:
            relative_affine = np.zeros((4, 4))
            relative_affine[3, 3] = 1

        gripper_action = None
        if self.teleop_mode == "robot":
            if self.continuous_gripper:
                # Initialize or update last gripper state
                if not hasattr(self, 'last_gripper_state'):
                    self.last_gripper_state = self.gripper_state
                
                # Calculate gripper change based on trigger values
                gripper_change = 0
                if self.controller_state.right_index_trigger > 0.5:
                    # Gradually close (move towards 1)
                    gripper_change = 0.05 * GRIPPER_CLOSE
                elif self.controller_state.right_hand_trigger > 0.5:
                    # Gradually open (move towards -1)
                    gripper_change = 0.05 * GRIPPER_OPEN
                
                # Update gripper state with change, clamping between -1 and 1
                gripper_action = np.clip(self.last_gripper_state + gripper_change, GRIPPER_OPEN, GRIPPER_CLOSE)
                self.last_gripper_state = gripper_action
            else:
                if self.controller_state.right_index_trigger > 0.5:
                    gripper_action = GRIPPER_CLOSE
                elif self.controller_state.right_hand_trigger > 0.5:
                    gripper_action = GRIPPER_OPEN

        if gripper_action is not None and gripper_action != self.gripper_state:
            self.gripper_state = gripper_action

        if self.start_teleop:
            relative_pos, relative_rot = (
                relative_affine[:3, 3],
                relative_affine[:3, :3],
            )

            target_pos = self.home_pos + relative_pos
            # Mirror the home rotation for left arm before combining with relative rotation
            if "deoxys_left" in self.deoxys_config_path:
                home_rot_mirrored = self.home_rot.copy()
                home_rot_mirrored[:2, :] *= -1
                target_rot = home_rot_mirrored @ relative_rot
            else:
                target_rot = self.home_rot @ relative_rot
            target_quat = transform_utils.mat2quat(target_rot)

            target_pos = np.clip(
                target_pos,
                a_min=ROBOT_WORKSPACE_MIN,
                a_max=ROBOT_WORKSPACE_MAX,
            )

        else:
            target_pos, target_quat = (
                self.home_pos + self.home_offset,
                transform_utils.mat2quat(self.home_rot),
            )

        action = FrankaAction(
            pos=target_pos.flatten().astype(np.float32),
            quat=target_quat.flatten().astype(np.float32),
            gripper=self.gripper_state,
            reset=False,
            timestamp=time.time(),
        )
        if self.teleop_mode == "robot":
            self.action_socket.send(bytes(pickle.dumps(action, protocol=-1)))
        else:
            self.action_socket.send(b"get_state")

        # self.action_socket.send(bytes(pickle.dumps(action, protocol=-1)))
        robot_state = self.action_socket.recv()

        robot_state = pickle.loads(robot_state)
        robot_state.start_teleop = self.start_teleop
        # self.state_socket.send(bytes(pickle.dumps(robot_state, protocol=-1)))
        self.state_socket.pub_keypoints(robot_state, "robot_state")
        self.commanded_state_socket.pub_keypoints(action, "commanded_robot_state")

    def stream(self):
        notify_component_start("Franka teleoperator control")
        print("Start controlling the robot hand using the Oculus Headset.\n")

        try:
            while True:
                # Retargeting function
                self._apply_retargeted_angles()
        except KeyboardInterrupt:
            pass
        finally:
            self._controller_state_subscriber.stop()
            self.action_socket.close()

        print("Stopping the teleoperator!")


def main():
    operator = FrankaOperator(save_states=True)
    operator.stream()


if __name__ == "__main__":
    main()
