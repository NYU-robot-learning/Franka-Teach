from typing import Dict, List, Optional
import cv2
import gym
import numpy as np
import time
import pickle
from concurrent.futures import ThreadPoolExecutor

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


class BimanualFrankaEnv(gym.Env):
    def __init__(
        self,
        robot_repr: Dict[str, str],
        cam_ids: List[int] = [1, 2, 3, 4, 51],
        width: int = 640,
        height: int = 480,
        use_robot: bool = True,
        sensor_type: Optional[str] = None,
        sensor_params: Optional[Dict] = None,
    ):
        super(BimanualFrankaEnv, self).__init__()
        self.width = width
        self.height = height
        self.channels = 3
        self.feature_dim = 8 * 2  # 8 features for each arm (pos, quat, gripper)
        # Double the action dimension for two arms
        self.action_dim = 14  # (pos, axis angle, gripper) for each arm

        self.use_robot = use_robot
        self.sensor_type = sensor_type
        self.sensor_params = sensor_params

        # Store robot representations
        self.left_robot_repr = robot_repr["left"]
        self.right_robot_repr = robot_repr["right"]

        self.n_channels = 3
        self.reward = 0

        self.left_franka_state = None
        self.right_franka_state = None
        self.curr_images = None

        # Double the action space for two arms
        self.action_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(self.action_dim,)
        )

        # Create observation space for both arms
        obs_space = {
            f"pixels{cam_id}": gym.spaces.Box(
                low=0, high=255, shape=(height, width, self.n_channels), dtype=np.uint8
            )
            for cam_id in cam_ids
        }

        # Features and proprioceptive for both arms
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
            for arm in ["left", "right"]:
                for sensor_idx in range(2):  # 2 sensors per arm
                    obs_space[f"{arm}_sensor{sensor_idx}"] = gym.spaces.Box(
                        low=-float("inf"),
                        high=float("inf"),
                        shape=(15,),  # sensor_dim
                        dtype=np.float32,
                    )
                    obs_space[f"{arm}_sensor{sensor_idx}_diffs"] = gym.spaces.Box(
                        low=-float("inf"),
                        high=float("inf"),
                        shape=(15,),
                        dtype=np.float32,
                    )

        self.observation_space = gym.spaces.Dict(obs_space)

        if self.use_robot:
            # Initialize camera subscribers
            self.image_subscribers = {}
            for cam_idx in cam_ids:
                port = CAM_PORT + cam_idx
                self.image_subscribers[cam_idx] = ZMQCameraSubscriber(
                    host=HOST,
                    port=port,
                    topic_type="RGB",
                )

            # Initialize control sockets for both arms
            self.left_action_socket = create_request_socket(
                HOST, PORTS[self.left_robot_repr]["control"]
            )
            self.right_action_socket = create_request_socket(
                HOST, PORTS[self.right_robot_repr]["control"]
            )

            if self.sensor_type == "reskin":
                self.left_sensor_subscriber = ReskinSensorSubscriber(
                    port=PORTS[self.left_robot_repr]["reskin"]
                )
                self.right_sensor_subscriber = ReskinSensorSubscriber(
                    port=PORTS[self.right_robot_repr]["reskin"]
                )
                self.sensor_prev_state = {
                    "left": None,
                    "right": None,
                }
                self.subtract_sensor_baseline = sensor_params[
                    "subtract_sensor_baseline"
                ]
                self.sensor_baseline = {}
                self._get_reskin_state(update_baseline=True)

    @staticmethod
    def _send_action_to_arm(action, socket, reset=False):
        pos = action[:3]
        quat = action[3:7]
        gripper = action[-1]
        if gripper < 0.0:
            gripper = GRIPPER_OPEN
        else:
            gripper = GRIPPER_CLOSE

        franka_action = FrankaAction(
            pos=pos,
            quat=quat,
            gripper=gripper,
            reset=reset,
            timestamp=time.time(),
        )

        socket.send(bytes(pickle.dumps(franka_action, protocol=-1)))
        return pickle.loads(socket.recv())

    def step(self, action):
        # Split action into left and right arm actions
        left_action = action[:7]
        right_action = action[7:]

        # Send actions to both arms in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(
                self._send_action_to_arm, left_action, self.left_action_socket
            )
            right_future = executor.submit(
                self._send_action_to_arm, right_action, self.right_action_socket
            )

            self.left_franka_state = left_future.result()
            self.right_franka_state = right_future.result()

        # Get camera images
        image_dict = {}
        self.curr_images = []
        for cam_id, subscriber in self.image_subscribers.items():
            image, _ = subscriber.recv_rgb_image()
            image_dict[f"pixels{cam_id}"] = cv2.resize(image, (self.width, self.height))
            self.curr_images.append(image)

        # Construct observation dictionary
        obs = {
            "features": np.concatenate(
                (
                    self.left_franka_state.pos,
                    self.left_franka_state.quat,
                    [self.left_franka_state.gripper],
                    self.right_franka_state.pos,
                    self.right_franka_state.quat,
                    [self.right_franka_state.gripper],
                )
            ),
            "proprioceptive": np.concatenate(
                (
                    self.left_franka_state.pos,
                    self.left_franka_state.quat,
                    [self.left_franka_state.gripper],
                    self.right_franka_state.pos,
                    self.right_franka_state.quat,
                    [self.right_franka_state.gripper],
                )
            ),
        }

        if self.sensor_type == "reskin":
            try:
                left_reskin_state = self._get_reskin_state(
                    self.left_sensor_subscriber, "left"
                )
                right_reskin_state = self._get_reskin_state(
                    self.right_sensor_subscriber, "right"
                )
                obs.update(left_reskin_state)
                obs.update(right_reskin_state)
            except KeyError:
                pass

        obs.update(image_dict)
        return obs, self.reward, False, False, {}

    def reset(self):
        print("resetting bimanual environment")

        # Reset both arms to a neutral position
        franka_reset_action = np.zeros(self.action_dim, dtype=np.float32)
        franka_reset_action[-1] = GRIPPER_OPEN  # Open gripper for reset

        # Send reset commands to both arms in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(
                self._send_action_to_arm,
                franka_reset_action,
                self.left_action_socket,
                reset=True,
            )
            right_future = executor.submit(
                self._send_action_to_arm,
                franka_reset_action,
                self.right_action_socket,
                reset=True,
            )

            left_franka_state: FrankaState = left_future.result()
            right_franka_state: FrankaState = right_future.result()

            self.left_franka_state = left_franka_state
            self.right_franka_state = right_franka_state

        # Get camera images
        image_dict = {}
        self.curr_images = []
        for cam_id, subscriber in self.image_subscribers.items():
            image, _ = subscriber.recv_rgb_image()
            image_dict[f"pixels{cam_id}"] = cv2.resize(image, (self.width, self.height))
            self.curr_images.append(image)

        # Construct observation dictionary
        obs = {
            "features": np.concatenate(
                (
                    left_franka_state.pos,
                    left_franka_state.quat,
                    [left_franka_state.gripper],
                    right_franka_state.pos,
                    right_franka_state.quat,
                    [right_franka_state.gripper],
                )
            ),
            "proprioceptive": np.concatenate(
                (
                    left_franka_state.pos,
                    left_franka_state.quat,
                    [left_franka_state.gripper],
                    right_franka_state.pos,
                    right_franka_state.quat,
                    [right_franka_state.gripper],
                )
            ),
        }

        if self.sensor_type == "reskin":
            try:
                left_reskin_state = self._get_reskin_state(
                    self.left_sensor_subscriber, "left", update_baseline=True
                )
                right_reskin_state = self._get_reskin_state(
                    self.right_sensor_subscriber, "right", update_baseline=True
                )
                obs.update(left_reskin_state)
                obs.update(right_reskin_state)
            except KeyError:
                pass

        obs.update(image_dict)
        return obs

    def _get_reskin_state(self, sensor_subscriber, arm_prefix, update_baseline=False):
        sensor_state = sensor_subscriber.get_sensor_state()
        sensor_values = np.array(sensor_state["sensor_values"], dtype=np.float32)

        if update_baseline:
            baseline_meas = []
            while len(baseline_meas) < 5:
                sensor_state = sensor_subscriber.get_sensor_state()
                sensor_values = np.array(
                    sensor_state["sensor_values"], dtype=np.float32
                )
                baseline_meas.append(sensor_values)
            self.sensor_baseline[arm_prefix] = np.mean(baseline_meas, axis=0)
            if self.subtract_sensor_baseline:
                self.sensor_prev_state[arm_prefix] = (
                    sensor_values - self.sensor_baseline[arm_prefix]
                )
            else:
                self.sensor_prev_state[arm_prefix] = sensor_values

        if self.subtract_sensor_baseline:
            sensor_values = sensor_values - self.sensor_baseline[arm_prefix]

        sensor_diff = sensor_values - self.sensor_prev_state[arm_prefix]
        self.sensor_prev_state[arm_prefix] = sensor_values

        reskin_state = {}
        for sensor_idx in range(2):  # 2 sensors per arm
            reskin_state[f"{arm_prefix}_sensor{sensor_idx}"] = sensor_values[
                sensor_idx * 15 : (sensor_idx + 1) * 15
            ]
            reskin_state[f"{arm_prefix}_sensor{sensor_idx}_diffs"] = sensor_diff[
                sensor_idx * 15 : (sensor_idx + 1) * 15
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
    # Example usage
    robot_repr = {
        "left": "deoxys_left",
        "right": "deoxys_right",
    }
    env = BimanualFrankaEnv(robot_repr=robot_repr, cam_ids=[1, 2])
    obs = env.reset()
    print("Initial Observation:", obs)
    import ipdb

    ipdb.set_trace()  # Debugging breakpoint
