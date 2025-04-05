import numpy as np
import os
# Read the hand information from the environment variable.
HAND = os.environ.get("HAND", "right")  # defaults to "right" if not set

# Decide the control port based on the hand.
if HAND == "left":
    CONTROL_PORT = 8911
    H_R_V = np.array([
        [-1,  0,  0,  0],
        [ 0,  0,  1,  0],
        [ 0, 1,  0,  0],
        [ 0,  0,  0,  1]
    ])
    H_R_V_star = np.array([
        [ 1,  0,  0,  0],
        [ 0,  0,  1,  0],
        [ 0, 1,  0,  0],
        [ 0,  0,  0,  1]
    ]) #working
elif HAND == "right":
    CONTROL_PORT = 8901
    H_R_V = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    H_R_V_star = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
else:
    CONTROL_PORT = 8901
    H_R_V = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    H_R_V_star = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
# Network constants
HOST = "localhost"
CAM_PORT = 10005
VR_CONTROLLER_STATE_PORT = 8889
STATE_PORT = 8900
# CONTROL_PORT = 8901
COMMANDED_STATE_PORT = 8902
RESKIN_STREAM_PORT = 12005


STATE_TOPIC = "state"
CONTROL_TOPIC = "control"

# VR constants
VR_TCP_HOST = "10.19.189.139"
VR_TCP_PORT = 5555
VR_CONTROLLER_TOPIC = b"oculus_controller"

# Robot constants
GRIPPER_OPEN = -1
GRIPPER_CLOSE = 1


# FRANKA_SERVER_CONFIG_PATH = "/home/bobby/Point-Policy/Franka-Teach/configs/franka_server.yaml"
# import yaml

# # Load and parse the YAML file
# with open(FRANKA_SERVER_CONFIG_PATH, 'r') as f:
#     config = yaml.safe_load(f)

# # Access the value of 'deoxys_config_path'
# deoxys_config = config.get("deoxys_config_path")
# if deoxys_config == "deoxys_right.yml":
#     H_R_V = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
#     H_R_V_star = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
# elif deoxys_config == "deoxys_left.yml":
#     H_R_V = np.array([
#         [-1,  0,  0,  0],
#         [ 0,  0,  1,  0],
#         [ 0, 1,  0,  0],
#         [ 0,  0,  0,  1]
#     ])
#     H_R_V_star = np.array([
#         [ 1,  0,  0,  0],
#         [ 0,  0,  1,  0],
#         [ 0, 1,  0,  0],
#         [ 0,  0,  0,  1]
#     ]) #working

x_min, x_max = 0.2, 0.75
y_min, y_max = -0.4, 0.4
z_min, z_max = 0.05, 0.7  # 232, 550
ROBOT_WORKSPACE_MIN = np.array([x_min, y_min, z_min])
ROBOT_WORKSPACE_MAX = np.array([x_max, y_max, z_max])

TRANSLATIONAL_POSE_VELOCITY_SCALE = 5
ROTATIONAL_POSE_VELOCITY_SCALE = 0.75
ROTATION_VELOCITY_LIMIT = 0.5
TRANSLATION_VELOCITY_LIMIT = 1

# Frequencies
# TODO: Separate VR and deploy frequencies
VR_FREQ = 20
CONTROL_FREQ = 20
STATE_FREQ = 100
CAM_FPS = 30
DEPTH_PORT_OFFSET = 1000
