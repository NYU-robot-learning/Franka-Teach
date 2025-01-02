import os
from pathlib import Path
import pickle
import time
import numpy as np

from deoxys.utils import YamlConfig
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import transform_utils
from deoxys.utils.config_utils import (
    get_default_controller_config,
    verify_controller_config,
)

from frankateach.utils import notify_component_start
from frankateach.network import create_response_socket
from frankateach.messages import FrankaAction, FrankaState
from frankateach.constants import (
    CONTROL_PORT,
    HOST,
    VR_FREQ,
)

CONFIG_ROOT = Path(__file__).parent / "configs"


class FrankaServer:
    def __init__(self, cfg):
        self._robot = Robot(cfg, VR_FREQ)
        # Action REQ/REP
        self.action_socket = create_response_socket(HOST, CONTROL_PORT)

    def init_server(self):
        # connect to robot
        print("Starting Franka server...")
        self._robot.reset_robot()
        self.control_daemon()

    def get_state(self):
        quat, pos = self._robot.last_eef_quat_and_pos
        gripper = self._robot.last_gripper_action
        if quat is not None and pos is not None and gripper is not None:
            state = FrankaState(
                pos=pos.flatten().astype(np.float32),
                quat=quat.flatten().astype(np.float32),
                gripper=gripper,
                timestamp=time.time(),
            )
            return bytes(pickle.dumps(state, protocol=-1))
        else:
            return b"state_error"

    def control_daemon(self):
        notify_component_start(component_name="Franka Control Subscriber")
        try:
            while True:
                command = self.action_socket.recv()
                if command == b"get_state":
                    self.action_socket.send(self.get_state())
                else:
                    franka_control: FrankaAction = pickle.loads(command)
                    if franka_control.reset:
                        self._robot.reset_joints(gripper_open=franka_control.gripper)
                        time.sleep(1)
                    else:
                        self._robot.osc_move(
                            franka_control.pos,
                            franka_control.quat,
                            franka_control.gripper,
                        )
                    time.sleep(0.1)
                    self.action_socket.send(self.get_state())
        except KeyboardInterrupt:
            pass
        finally:
            self._robot.close()
            self.action_socket.close()


class Robot(FrankaInterface):
    def __init__(self, cfg, control_freq):
        super(Robot, self).__init__(
            general_cfg_file=os.path.join(CONFIG_ROOT, cfg),
            use_visualizer=False,
            control_freq=control_freq,
        )
        self.velocity_controller_cfg = verify_controller_config(
            YamlConfig(
                os.path.join(CONFIG_ROOT, "osc-pose-controller.yml")
            ).as_easydict()
        )

    def reset_robot(self):
        self.reset()

        print("Waiting for the robot to connect...")
        while len(self._state_buffer) == 0:
            time.sleep(0.01)

        print("Franka is connected")

    def osc_move(self, target_pos, target_quat, gripper_state):
        target_mat = transform_utils.pose2mat(pose=(target_pos, target_quat))

        idx = 0
        for idx in range(80):
            current_quat, current_pos = self.last_eef_quat_and_pos
            current_mat = transform_utils.pose2mat(
                pose=(current_pos.flatten(), current_quat.flatten())
            )

            pose_error = transform_utils.get_pose_error(
                target_pose=target_mat, current_pose=current_mat
            )
            # if np.linalg.norm(pose_error[:3]) < 0.01:
            #     break

            if np.dot(target_quat, current_quat) < 0.0:
                current_quat = -current_quat

            quat_diff = transform_utils.quat_distance(target_quat, current_quat)
            axis_angle_diff = transform_utils.quat2axisangle(quat_diff)

            # action_pos = pose_error[:3]
            # action_axis_angle = axis_angle_diff.flatten()
            action_pos = pose_error[:3] * 10
            action_axis_angle = axis_angle_diff.flatten() * 1
            action_pos = np.clip(action_pos, -1.0, 1.0)
            action_axis_angle = np.clip(action_axis_angle, -0.5, 0.5)

            action = action_pos.tolist() + action_axis_angle.tolist() + [gripper_state]

            self.control(
                controller_type="OSC_POSE",
                action=action,
                controller_cfg=self.velocity_controller_cfg,
            )
        print(idx)

        # target_axis_angle = transform_utils.quat2axisangle(target_quat)
        # current_rot, current_pos = self.last_eef_rot_and_pos

        # for _ in range(80):
        #     current_pose = self.last_eef_pose
        #     current_pos = current_pose[:3, 3:]
        #     current_rot = current_pose[:3, :3]
        #     current_quat = transform_utils.mat2quat(current_rot)
        #     if np.dot(target_quat, current_quat) < 0.0:
        #         current_quat = -current_quat
        #     quat_diff = transform_utils.quat_distance(target_quat, current_quat)
        #     current_axis_angle = transform_utils.quat2axisangle(current_quat)
        #     axis_angle_diff = transform_utils.quat2axisangle(quat_diff)
        #     action_pos = (target_pos - current_pos).flatten() * 10
        #     action_axis_angle = axis_angle_diff.flatten() * 1
        #     action_pos = np.clip(action_pos, -1.0, 1.0)
        #     action_axis_angle = np.clip(action_axis_angle, -0.5, 0.5)

        #     action = action_pos.tolist() + action_axis_angle.tolist() + [gripper_state]
        #     # logger.info(f"Axis angle action {action_axis_angle.tolist()}")
        #     # print(np.round(action, 2))
        #     self.control(
        #         controller_type="OSC_POSE",
        #         action=action,
        #         controller_cfg=self.velocity_controller_cfg,
        #     )

    def reset_joints(
        self,
        timeout=7,
        gripper_open=False,
    ):
        start_joint_pos = [
            0.09162008114028396,
            -0.19826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.30396583422025,
            0.8480939705504309,
        ]
        assert type(start_joint_pos) is list or type(start_joint_pos) is np.ndarray
        controller_cfg = get_default_controller_config(controller_type="JOINT_POSITION")

        if gripper_open:
            gripper_action = -1
        else:
            gripper_action = 1

        # This is for varying initialization of joints a little bit to
        # increase data variation.
        # start_joint_pos = [
        #     e + np.clip(np.random.randn() * 0.005, -0.005, 0.005)
        #     for e in start_joint_pos
        # ]
        if type(start_joint_pos) is list:
            action = start_joint_pos + [gripper_action]
        else:
            action = start_joint_pos.tolist() + [gripper_action]
        start_time = time.time()
        while True:
            if self.received_states and self.check_nonzero_configuration():
                if (
                    np.max(np.abs(np.array(self.last_q) - np.array(start_joint_pos)))
                    < 1e-3
                ):
                    break
            self.control(
                controller_type="JOINT_POSITION",
                action=action,
                controller_cfg=controller_cfg,
            )
            end_time = time.time()

            # Add timeout
            if end_time - start_time > timeout:
                break
        return True
