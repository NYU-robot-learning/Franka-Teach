import cv2
import gym
import numpy as np
import time
import pickle
from typing import Dict, List, Optional

from frankateach.constants import (
    CAM_PORT,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    HOST,
    INTERNET_HOST,
    PORTS,
    K, 
    D, 
    T_robot_to_camera, 
    T_aruco_to_camera,
    left_camera_calibs,
    FRANKA_INITIAL_POS,
    FRANKA_INITIAL_QUAT,
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


class FrankaEnv(gym.Env):
    def __init__(
        self,
        robot_repr: Optional[Dict] = None,
        cam_ids: List[int] = [1, 2, 3, 4, 51, 6],  # 6 is iphone
        width: int = 640,
        height: int = 480,
        use_robot: bool = True,
        use_gt_depth=False,
        sensor_type: Optional[str] = None,
        sensor_params: Optional[Dict] = None,
        crop_h: Optional[int]=None,
        crop_w: Optional[int]=None,
    ):
        super(FrankaEnv, self).__init__()
        self.width = width
        self.height = height
        self.crop_h = crop_h
        self.crop_w = crop_w

        self.feature_dim = 8
        self.action_dim = 7  # (pos, axis angle, gripper)
        self.use_robot = use_robot
        self.use_gt_depth = use_gt_depth
        self.sensor_type = sensor_type
        
        self.n_channels = 3
        self.reward = 0

        self.franka_state = None
        self.curr_images = None

        if sensor_type is not None:
            assert sensor_type in ["reskin"]
            assert (
                ReskinSensorSubscriber is not None
            ), "ReskinSensorSubscriber not found"
            if sensor_type == "reskin":
                self.n_sensors = 2
                self.sensor_dim = 15
        self.sensor_params = sensor_params

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
            if self.use_gt_depth:
                self.depth_subscribers = {}
            for cam_id in cam_ids:
                port = CAM_PORT + cam_id
                self.image_subscribers[cam_id] = ZMQCameraSubscriber(
                    host=INTERNET_HOST if cam_id == 6 else HOST,
                    port=port,
                    topic_type="RGB",
                )

                if self.use_gt_depth:
                    depth_port = CAM_PORT + cam_id + 1000  # depth offset =1000
                    self.depth_subscribers[cam_id] = ZMQCameraSubscriber(
                        host=INTERNET_HOST if cam_id == 6 else HOST,
                        port=depth_port,
                        topic_type="Depth",
                    )

            if self.sensor_type == "reskin":
                self.sensor_subscriber = ReskinSensorSubscriber(
                    port=PORTS[robot_repr]["reskin"]
                )

                self.sensor_prev_state = None
                self.subtract_sensor_baseline = sensor_params[
                    "subtract_sensor_baseline"
                ]

                # Call once to populate initial baseline
                self._get_reskin_state(update_baseline=True)

            self.action_request_socket = create_request_socket(
                HOST, PORTS[robot_repr]["control"]
            )

    def get_state(self):
        self.action_request_socket.send(b"get_state")
        franka_state: FrankaState = pickle.loads(self.action_request_socket.recv())
        self.franka_state = franka_state
        return franka_state

    def step(self, abs_action):
        pos = abs_action[:3]
        quat = abs_action[3:7]
        gripper = abs_action[-1]
        if gripper < 0.0:
            gripper = GRIPPER_OPEN
        else:
            gripper = GRIPPER_CLOSE

        # Send action to the robot
        franka_action = FrankaAction(
            pos=pos,
            quat=quat,
            gripper=gripper,
            reset=False,
            timestamp=time.time(),
        )
        # print("sending action to robot: ", franka_action)

        self.action_request_socket.send(bytes(pickle.dumps(franka_action, protocol=-1)))
        franka_state: FrankaState = pickle.loads(self.action_request_socket.recv())
        self.franka_state = franka_state

        image_dict = {}
        for cam_id, subscriber in self.image_subscribers.items():
            image, _ = subscriber.recv_rgb_image()
            # crop the image
            if self.crop_h is not None and self.crop_w is not None:
                h, w, _ = image.shape
                image = image[
                    int(h * self.crop_h[0]) : int(h * self.crop_h[1]),
                    int(w * self.crop_w[0]) : int(w * self.crop_w[1]),
                ]
            image_dict[f"pixels{cam_id}"] = cv2.resize(image, (self.width, self.height))
        self.curr_images = image_dict

        if self.use_gt_depth:
            depth_dict = {}
            for cam_id, subscriber in self.depth_subscribers.items():
                depth, _ = subscriber.recv_depth_image()

                if self.crop_h is not None and self.crop_w is not None:
                    h, w = depth.shape
                    depth = depth[
                        int(h * self.crop_h[0]) : int(h * self.crop_h[1]),
                        int(w * self.crop_w[0]) : int(w * self.crop_w[1]),
                    ]

                depth_dict[f"depth{cam_id}"] =  depth_dict[f"depth{cam_id}"] = cv2.resize(depth, (self.width, self.height))

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
        if self.use_gt_depth:
            obs.update(depth_dict)
        return obs, self.reward, False, False, {}

    def reset(self):
        if self.use_robot:
            print("resetting")
            # TODO: send b"reset" to the robot instead of this action
            franka_reset_action = FrankaAction(
                pos=FRANKA_INITIAL_POS,
                quat=FRANKA_INITIAL_QUAT,
                gripper=GRIPPER_OPEN,
                reset=True,
                timestamp=time.time(),
            )

            self.action_request_socket.send(
                bytes(pickle.dumps(franka_reset_action, protocol=-1))
            )
            franka_state: FrankaState = pickle.loads(self.action_request_socket.recv())
            self.franka_state = franka_state
            print("reset done: ", franka_state)

            image_dict = {}
            for cam_id, subscriber in self.image_subscribers.items():
                image, _ = subscriber.recv_rgb_image()
                # crop the image
                if self.crop_h is not None and self.crop_w is not None:
                    h, w, _ = image.shape
                    image = image[
                        int(h * self.crop_h[0]) : int(h * self.crop_h[1]),
                        int(w * self.crop_w[0]) : int(w * self.crop_w[1]),
                    ]
                image_dict[f"pixels{cam_id}"] = cv2.resize(image, (self.width, self.height))
            self.curr_images = image_dict

            if self.use_gt_depth:
                depth_dict = {}
                for cam_id, subscriber in self.depth_subscribers.items():
                    depth, _ = subscriber.recv_depth_image()

                    if self.crop_h is not None and self.crop_w is not None:
                        h, w = depth.shape
                        depth = depth[
                            int(h * self.crop_h[0]) : int(h * self.crop_h[1]),
                            int(w * self.crop_w[0]) : int(w * self.crop_w[1]),
                        ]

                    depth_dict[f"depth{cam_id}"] = cv2.resize(depth, (self.width, self.height))

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
            if self.use_gt_depth:
                obs.update(depth_dict)

        else:
            obs = {}
            obs["features"] = np.zeros(self.feature_dim)
            obs["pixels"] = np.zeros((self.height, self.width, self.n_channels))
            obs["pixels6"] = np.zeros((self.height, self.width, self.n_channels))
            obs["depth6"] = np.zeros((self.height, self.width))
            if self.use_gt_depth:
                obs["depth"] = np.zeros((self.height, self.width))

        return obs

    def _get_reskin_state(self, update_baseline=False):
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

    def render(self, mode="rgb_array", cam_id=None, width=640, height=480):
        assert self.curr_images is not None, "Must call reset() before render()"
        if mode == "rgb_array":
            if cam_id is not None:
                return self.curr_images[f"pixels{cam_id}"]
            image_list = []
            for key, im in self.curr_images.items():
                h, w = im.shape[:2]
                aspect_ratio = w / h
                new_height = height  # desired height
                new_width = int(aspect_ratio * new_height)
                image_list.append(cv2.resize(im, (new_width, new_height)))

            return np.concatenate(image_list, axis=1)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    env = FrankaEnv()
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
