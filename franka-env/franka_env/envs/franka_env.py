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
    # CONTROL_PORT,
    CONTROL_PORT_LEFT,
    CONTROL_PORT_RIGHT,
    DEPTH_PORT_OFFSET,
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
        cam_ids=[1, 2, 3, 4, 51],
        zed_ids=None,
        width=640,
        height=480,
        use_robot=True,
        use_gt_depth=False,
        sensor_type=None,
        sensor_params=None,
        side="right",
    ):
        super(FrankaEnv, self).__init__()
        self.width = width
        self.height = height
        self.channels = 3
        self.feature_dim = 8
        self.action_dim = 7  # (pos, axis angle, gripper)

        self.use_robot = use_robot
        self.use_gt_depth = use_gt_depth
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
        self.zed_ids = zed_ids

        self.n_channels = 3
        self.reward = 0

        self.franka_state = None
        self.curr_images = None

        self.action_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(self.action_dim,)
        )
        
        obs_space = {}
        for cam_id in cam_ids:
            if zed_ids is not None and cam_id in zed_ids:
                obs_space[f"pixels{cam_id}_left"] = gym.spaces.Box(
                    low=0, high=255, shape=(height, width, self.n_channels), dtype=np.uint8
                )
                obs_space[f"pixels{cam_id}_right"] = gym.spaces.Box(
                    low=0, high=255, shape=(height, width, self.n_channels), dtype=np.uint8
                )
            else:
                obs_space[f"pixels{cam_id}"] = gym.spaces.Box(
                    low=0, high=255, shape=(height, width, self.n_channels), dtype=np.uint8
                )
                if self.use_gt_depth:
                    obs_space[f"depth{cam_id}"] = gym.spaces.Box(
                        low=0, high=255, shape=(height, width), dtype=np.uint8
                    )
        
        obs_space["features"] = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(self.feature_dim,),
            dtype=np.float32,
        )
        # obs_space["proprioceptive"] = gym.spaces.Box(
        #     low=-float("inf"),
        #     high=float("inf"),
        #     shape=(self.feature_dim,),
        #     dtype=np.float32,
        # )
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

        # if self.use_robot:
        self.image_subscribers = {}
        self.depth_subscribers = {}
        for cam_idx in cam_ids:
            port = CAM_PORT + cam_idx
            self.image_subscribers[cam_idx] = ZMQCameraSubscriber(
                host=HOST,
                port=port,
                topic_type="RGB",
            )
            if self.use_gt_depth:
                self.depth_subscribers[cam_idx] = ZMQCameraSubscriber(
                    host=HOST,
                    port=port + DEPTH_PORT_OFFSET,
                    topic_type="Depth",
                )

        if self.sensor_type == "reskin":
            self.sensor_subscriber = ReskinSensorSubscriber()

            self.sensor_prev_state = None
            self.subtract_sensor_baseline = sensor_params[
                "subtract_sensor_baseline"
            ]

            # Call once to populate initial baseline
            self._get_reskin_state(update_baseline=True)

        self.action_request_socket = create_request_socket(
            HOST, CONTROL_PORT_RIGHT if side == "right" else CONTROL_PORT_LEFT
        )

    def get_state(self):
        if self.use_robot:
            self.action_request_socket.send(b"get_state")
            franka_state: FrankaState = pickle.loads(self.action_request_socket.recv())
        else:
            franka_state = FrankaState(
                pos=np.zeros(3),
                quat=np.zeros(4),
                gripper=GRIPPER_OPEN,
                timestamp=time.time(),
            )
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
        
        # if gripper < -0.8 and self.prev_gripper == GRIPPER_CLOSE:
        #     gripper = -1.0
        # elif gripper > 0.5 and self.prev_gripper == GRIPPER_OPEN:
        #     gripper = 1.0
        # else:
        #     gripper = self.prev_gripper
        self.prev_gripper = gripper

        # Send action to the robot
        franka_action = FrankaAction(
            pos=pos,
            quat=quat,
            gripper=gripper,
            reset=False,
            timestamp=time.time(),
        )
        # print("sending action to robot: ", franka_action)

        # self.action_request_socket.send(bytes(pickle.dumps(franka_action, protocol=-1)))
        # franka_state: FrankaState = pickle.loads(self.action_request_socket.recv())
        if self.use_robot:
            self.action_request_socket.send(bytes(pickle.dumps(franka_action, protocol=-1)))
            franka_state: FrankaState = pickle.loads(self.action_request_socket.recv())
        else:
            franka_state = FrankaState(
                pos=np.zeros(3),
                quat=np.zeros(4),
                gripper=GRIPPER_OPEN,
                timestamp=time.time(),
            )
        self.franka_state = franka_state

        image_dict = {}
        self.curr_images = []
        for cam_id, subscriber in self.image_subscribers.items():
            if self.use_robot:
                image, _ = subscriber.recv_rgb_image()
            else:
                image = np.zeros((self.height, self.width, 3), dtype=np.uint8) if cam_id not in self.zed_ids else np.zeros((self.height, self.width*2, 3), dtype=np.uint8)
            if self.zed_ids is not None and cam_id in self.zed_ids:
                h, w = image.shape[:2]
                left_image = image[:, : w // 2, :]
                right_image = image[:, w // 2 :, :]
                image_dict[f"pixels{cam_id}_left"] = left_image
                image_dict[f"pixels{cam_id}_right"] = right_image
            else:
                if image.shape[:2] != (self.height, self.width):
                    image = cv2.resize(image, (self.width, self.height))
                image_dict[f"pixels{cam_id}"] = image
            self.curr_images.append(image)

        if self.use_gt_depth:
            depth_dict = {}
            for cam_id, subscriber in self.depth_subscribers.items():
                if self.zed_ids is not None and cam_id in self.zed_ids:
                    # depth is not supported for ZED cameras
                    continue
                if self.use_robot:
                    depth, _ = subscriber.recv_depth_image()
                else:
                    depth = np.zeros((self.height, self.width), dtype=np.float32)
                if depth.shape[:2] != (self.height, self.width):
                    depth = cv2.resize(depth, (self.width, self.height))
                depth_dict[f"depth{cam_id}"] = depth

        obs = {
            "features": np.concatenate(
                (franka_state.pos, franka_state.quat, [franka_state.gripper])
            ),
            # "proprioceptive": np.concatenate(
            #     (franka_state.pos, franka_state.quat, [franka_state.gripper])
            # ),
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
        # for i, image in image_dict.items():
        #     obs[f"pixels{i}"] = cv2.resize(image, (self.width, self.height))
        return obs, self.reward, False, False, {}
        # return obs, self.reward, False, {}

    def reset(self, **kwargs):
        print("resetting")
        # TODO: send b"reset" to the robot instead of this action
        franka_reset_action = FrankaAction(
            pos=np.zeros(3),
            quat=np.zeros(4),
            gripper=GRIPPER_OPEN,
            reset=True,
            timestamp=time.time(),
        )
        self.prev_gripper = GRIPPER_OPEN

        if self.use_robot:
            self.action_request_socket.send(
                bytes(pickle.dumps(franka_reset_action, protocol=-1))
            )
            franka_state: FrankaState = pickle.loads(self.action_request_socket.recv())
        else:
            franka_state = FrankaState(
                pos=np.zeros(3),
                quat=np.zeros(4),
                gripper=GRIPPER_OPEN,
                timestamp=time.time(),
            )
        self.franka_state = franka_state
        print("reset done: ", franka_state)

        image_dict = {}
        self.curr_images = []
        for cam_id, subscriber in self.image_subscribers.items():
            if self.use_robot:
                image, _ = subscriber.recv_rgb_image()
            else:
                image = np.zeros((self.height, self.width, 3), dtype=np.uint8) if cam_id not in self.zed_ids else np.zeros((self.height, self.width*2, 3), dtype=np.uint8)
            if self.zed_ids is not None and cam_id in self.zed_ids:
                h, w = image.shape[:2]
                left_image = image[:, : w // 2, :]
                right_image = image[:, w // 2 :, :]
                image_dict[f"pixels{cam_id}_left"] = left_image
                image_dict[f"pixels{cam_id}_right"] = right_image
            else:
                if image.shape[:2] != (self.height, self.width):
                    image = cv2.resize(image, (self.width, self.height))
                image_dict[f"pixels{cam_id}"] = image
            self.curr_images.append(image)

        if self.use_gt_depth:
            depth_dict = {}
            for cam_id, subscriber in self.depth_subscribers.items():
                if self.zed_ids is not None and cam_id in self.zed_ids:
                    # depth is not supported for ZED cameras
                    continue
                if self.use_robot:
                    depth, _ = subscriber.recv_depth_image()
                else:
                    depth = np.zeros((self.height, self.width), dtype=np.float32)
                if depth.shape[:2] != (self.height, self.width):
                    depth = cv2.resize(depth, (self.width, self.height))
                depth_dict[f"depth{cam_id}"] = depth

        obs = {
            "features": np.concatenate(
                (franka_state.pos, franka_state.quat, [franka_state.gripper])
            ),
            # "proprioceptive": np.concatenate(
            #     (franka_state.pos, franka_state.quat, [franka_state.gripper])
            # ),
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
        # for i, image in enumerate(image_list):
        #     obs[f"pixels{i}"] = cv2.resize(image, (self.width, self.height))
        print("returning obs")
        return obs

    def _get_reskin_state(self, update_baseline=False):
        sensor_state = self.sensor_subscriber.get_sensor_state()
        sensor_values = np.array(sensor_state["sensor_values"], dtype=np.float32)
        if update_baseline:
            baseline_meas = []
            while len(baseline_meas) < 5:
                if self.use_robot:
                    sensor_state = self.sensor_subscriber.get_sensor_state()
                else:
                    sensor_state = {
                        "sensor_values": np.zeros(self.n_sensors * self.sensor_dim)
                    }
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
