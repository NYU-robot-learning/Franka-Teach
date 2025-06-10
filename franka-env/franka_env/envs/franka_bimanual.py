from typing import Dict, List, Optional, Union
import cv2
import gym
import numpy as np
import time
import pickle

from frankateach.constants import (
    CAM_PORT,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    HOST,
    PORTS,
)
from frankateach.messages import FrankaAction, FrankaState
from frankateach.network import (
    ZMQCameraSubscriber,
    create_request_socket,
)

try:
    from frankateach.sensors.reskin import ReskinSensorSubscriber
except ImportError:
    print("ReskinSensorSubscriber not found")
    ReskinSensorSubscriber = None


class FrankaBimanualEnv(gym.Env):
    def __init__(
        self,
        robot_repr: Union[str, List[str]],
        cam_ids: List[int] = [1, 2, 3, 4, 51],
        width: int = 640,
        height: int = 480,
        use_robot: bool = True,
        sensor_type: Optional[str] = None,
        sensor_params: Optional[Dict] = None,
    ):
        super(FrankaBimanualEnv, self).__init__()
        if isinstance(robot_repr, str):
            robot_repr = [robot_repr]
        self.width = width
        self.height = height
        self.channels = 3
        self.feature_dim = 8
        self.action_dim = 7  # (pos, axis angle, gripper)

        self.use_robot = use_robot
        self.sensor_type = sensor_type
        if sensor_type is not None:
            assert sensor_type in ["reskin"]
            assert (
                ReskinSensorSubscriber is not None
            ), "ReskinSensorSubscriber not found"
            if sensor_type == "reskin":
                self.n_sensors = 2
                self.sensor_dim = 15
        self.sensor_params = sensor_params

        self.n_channels = 3
        self.reward = 0

        self.franka_state = None
        self.curr_images = None

        self.action_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(self.action_dim,)
        )
        obs_space = {
            f"pixels{cam_id}": gym.spaces.Box(
                low=0, high=255, shape=(height, width, self.n_channels), dtype=np.uint8
            )
            for cam_id in cam_ids
        }
        obs_space["features"] = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(self.feature_dim,),
            dtype=np.float32,
        )
        obs_space["proprioceptive"] = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(self.feature_dim,),
            dtype=np.float32,
        )
        if self.sensor_type == "reskin":
            for sensor_idx in range(self.n_sensors):
                obs_space[f"sensor{sensor_idx}"] = gym.spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(self.sensor_dim,),
                    dtype=np.float32,
                )
                obs_space[f"sensor{sensor_idx}_diffs"] = gym.spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(self.sensor_dim,),
                    dtype=np.float32,
                )
        self.observation_space = gym.spaces.Dict(obs_space)

        if self.use_robot:
            self.image_subscribers = {}
            for cam_idx in cam_ids:
                port = CAM_PORT + cam_idx
                self.image_subscribers[cam_idx] = ZMQCameraSubscriber(
                    host=HOST,
                    port=port,
                    topic_type="RGB",
                )

            if self.sensor_type == "reskin":
                self.sensor_subscribers = {}
                for robot_id in robot_repr:
                    self.sensor_subscribers[robot_id] = ReskinSensorSubscriber(
                        port=PORTS[robot_id]["reskin"]
                    )

                self.sensor_prev_state = None
                self.subtract_sensor_baseline = sensor_params[
                    "subtract_sensor_baseline"
                ]

                # Call once to populate initial baseline
                self._get_reskin_state(update_baseline=True)

            self.action_request_sockets = {}
            for robot_id in robot_repr:
                # Create a request socket for each robot representation
                self.action_request_sockets[robot_id] = create_request_socket(
                    HOST, PORTS[robot_id]["control"]
                )
            self.franka_state = {}

    def get_state(self):
        for robot_id, socket in self.action_request_sockets.items():
            # Send request to get the state of the robot
            socket.send(b"get_state")
            franka_state: FrankaState = pickle.loads(socket.recv())
            self.franka_state[robot_id] = franka_state
        return self.franka_state

    def step(self, abs_action: Dict[str, np.ndarray]):
        """
        Step the environment with an absolute action.
        The action should be a dictionary with keys 'pos', 'quat', and 'gripper'.
        """
        # Send actions to each robot
        for robot_id, action_request_socket in self.action_request_sockets.items():
            franka_action = self._convert_action(abs_action[robot_id])
            action_request_socket.send(bytes(pickle.dumps(franka_action, protocol=-1)))

        # Receive the state of each robot after sending the action
        for robot_id, action_request_socket in self.action_request_sockets.items():
            franka_state: FrankaState = pickle.loads(action_request_socket.recv())
            self.franka_state[robot_id] = franka_state

        image_dict = {}
        self.curr_images = []
        for cam_id, subscriber in self.image_subscribers.items():
            image, _ = subscriber.recv_rgb_image()
            image_dict[f"pixels{cam_id}"] = cv2.resize(image, (self.width, self.height))
            self.curr_images.append(image)

        # TODO: handle multiple robots' states and images
        obs = {
            "features": np.concatenate(
                (franka_state.pos, franka_state.quat, [franka_state.gripper])
            ),
            "proprioceptive": np.concatenate(
                (franka_state.pos, franka_state.quat, [franka_state.gripper])
            ),
        }
        if self.sensor_type == "reskin":
            try:
                reskin_state = self._get_reskin_state()
                obs.update(reskin_state)
            except KeyError:
                pass

        obs.update(image_dict)
        # for i, image in image_dict.items():
        #     obs[f"pixels{i}"] = cv2.resize(image, (self.width, self.height))
        return obs, self.reward, False, False, {}

    def _convert_action(self, abs_action: np.ndarray):
        """
        Convert an absolute action to a FrankaAction.
        The action should be a numpy array of shape (7,).
        """
        if len(abs_action) != self.action_dim:
            raise ValueError(
                f"Action must have {self.action_dim} dimensions, got {len(abs_action)}"
            )

        pos = abs_action[:3]
        quat = abs_action[3:7]
        gripper = abs_action[-1]
        if gripper < 0.0:
            gripper = GRIPPER_OPEN
        else:
            gripper = GRIPPER_CLOSE

        return FrankaAction(
            pos=pos,
            quat=quat,
            gripper=gripper,
            reset=False,
            timestamp=time.time(),
        )

    def reset(self):
        print("resetting")
        # TODO: send b"reset" to the robot instead of this action
        franka_reset_action = FrankaAction(
            pos=np.zeros(3),
            quat=np.zeros(4),
            gripper=GRIPPER_OPEN,
            reset=True,
            timestamp=time.time(),
        )

        # Send reset action to each robot
        for robot_id, action_request_socket in self.action_request_sockets.items():
            print(f"Sending reset action to robot {robot_id}")
            # Send the reset action
            action_request_socket.send(
                bytes(pickle.dumps(franka_reset_action, protocol=-1))
            )
        # Receive the state of each robot after reset
        for robot_id, action_request_socket in self.action_request_sockets.items():
            print(f"Receiving state from robot {robot_id}")
            # Receive the state
            franka_state: FrankaState = pickle.loads(self.action_request_socket.recv())
            self.franka_state[robot_id] = franka_state
        print("reset done: ", franka_state)

        image_dict = {}
        self.curr_images = []
        for cam_id, subscriber in self.image_subscribers.items():
            image, _ = subscriber.recv_rgb_image()
            image_dict[f"pixels{cam_id}"] = cv2.resize(image, (self.width, self.height))
            self.curr_images.append(image)

        # TODO: handle multiple robots' states and images
        obs = {
            "features": np.concatenate(
                (franka_state.pos, franka_state.quat, [franka_state.gripper])
            ),
            "proprioceptive": np.concatenate(
                (franka_state.pos, franka_state.quat, [franka_state.gripper])
            ),
        }
        if self.sensor_type == "reskin":
            try:
                reskin_state = self._get_reskin_state(update_baseline=True)
                obs.update(reskin_state)
            except KeyError:
                pass

        obs.update(image_dict)
        # for i, image in enumerate(image_list):
        #     obs[f"pixels{i}"] = cv2.resize(image, (self.width, self.height))
        print("returning obs")
        return obs

    def _get_reskin_state(self, update_baseline=False):
        # TODO: make compatible with multiple robots
        sensor_state = self.sensor_subscriber.get_sensor_state()
        sensor_values = np.array(sensor_state["sensor_values"], dtype=np.float32)
        if update_baseline:
            baseline_meas = []
            while len(baseline_meas) < 5:
                sensor_state = self.sensor_subscriber.get_sensor_state()
                sensor_values = np.array(
                    sensor_state["sensor_values"], dtype=np.float32
                )
                baseline_meas.append(sensor_values)
            self.sensor_baseline = np.mean(baseline_meas, axis=0)
            if self.subtract_sensor_baseline:
                self.sensor_prev_state = sensor_values - self.sensor_baseline
            else:
                self.sensor_prev_state = sensor_values
        if self.subtract_sensor_baseline:
            sensor_values = sensor_values - self.sensor_baseline

        sensor_diff = sensor_values - self.sensor_prev_state
        self.sensor_prev_state = sensor_values
        sensor_keys = [f"sensor{sensor_idx}" for sensor_idx in range(self.n_sensors)]
        reskin_state = {}
        for sidx, sensor_key in enumerate(sensor_keys):
            reskin_state[sensor_key] = sensor_values[
                sidx * self.sensor_dim : (sidx + 1) * self.sensor_dim
            ]
            reskin_state[f"{sensor_key}_diffs"] = sensor_diff[
                sidx * self.sensor_dim : (sidx + 1) * self.sensor_dim
            ]
        return reskin_state

    def render(self, mode="rgb_array", width=640, height=480):
        assert self.curr_images is not None, "Must call reset() before render()"
        if mode == "rgb_array":
            image_list = []
            for im in self.curr_images:
                image_list.append(cv2.resize(im, (width, height)))

            return np.concatenate(image_list, axis=1)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    env = FrankaBimanualEnv()
    images = []
    obs = env.reset()

    apply_deltas = False
    if apply_deltas:
        delta_pos = 0.03
        delta_angle = 0.05
        for i in range(100):
            obs, reward, done, _ = env.step([delta_pos, 0, 0, 0, 0, 0, GRIPPER_OPEN])
            images.append(obs["pixels_0"])

        for i in range(100):
            obs, reward, done, _ = env.step([0, delta_pos, 0, 0, 0, 0, GRIPPER_OPEN])
            images.append(obs["pixels_0"])

        for i in range(100):
            obs, reward, done, _ = env.step([0, 0, delta_pos, 0, 0, 0, GRIPPER_OPEN])
            images.append(obs["pixels_0"])

        for i in range(100):
            obs, reward, done, _ = env.step([0, 0, 0, delta_angle, 0, 0, GRIPPER_OPEN])
            images.append(obs["pixels_0"])

        for i in range(100):
            obs, reward, done, _ = env.step([0, 0, 0, 0, delta_angle, 0, GRIPPER_OPEN])
            images.append(obs["pixels_0"])

        for i in range(100):
            obs, reward, done, _ = env.step([0, 0, 0, 0, 0, delta_angle, GRIPPER_OPEN])
            images.append(obs["pixels_0"])

        np.save("images.npy", np.array(images))
