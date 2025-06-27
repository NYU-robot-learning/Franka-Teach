import numpy as np

# Network constants
HOST = "localhost"
CAM_PORT = 10005
VR_CONTROLLER_STATE_PORT = 8889
INTERNET_HOST = "10.21.118.95" # computer(that run stream_iphone.py) IP address 

STATE_PORT = 8900
CONTROL_PORT = 8901
COMMANDED_STATE_PORT = 8902
RESKIN_STREAM_PORT = 12005

PORTS = {
    "deoxys_left": {
        "state": STATE_PORT,
        "control": CONTROL_PORT,
        "commanded_state": COMMANDED_STATE_PORT,
        "reskin": RESKIN_STREAM_PORT,
    },
    "deoxys_right": {
        "state": STATE_PORT + 50,
        "control": CONTROL_PORT + 50,
        "commanded_state": COMMANDED_STATE_PORT + 50,
        "reskin": RESKIN_STREAM_PORT + 50,
    },
}

STATE_TOPIC = "state"
CONTROL_TOPIC = "control"

# VR constants
VR_TCP_HOST = "10.19.189.139"
VR_TCP_PORT = 5555
VR_CONTROLLER_TOPIC = b"oculus_controller"

# Robot constants
GRIPPER_OPEN = -1
GRIPPER_CLOSE = 1
H_R_V = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
H_R_V_star = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
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

# Franka Initial position and orientation
FRANKA_INITIAL_POS = np.array([0.4579441, 0.0321529, 0.76579893])
FRANKA_INITIAL_QUAT=np.array([0.99984777, 0.00877362, 0.01497245, 0.00180537])


# Camera Constants Cage setting
left_camera_calibs = np.load(
    "/nas/projectaria/0331_calib_left_demo_7.npy",
    allow_pickle=True,
)[()]
right_camera_calibs = np.load(
    "/home/david/egozero/calib/calib_right_0321.npy",
    allow_pickle=True,
)[()]


K = {
    1: right_camera_calibs["cam_1"]["int"],
    2: right_camera_calibs["cam_2"]["int"],
    3: left_camera_calibs["cam_3"]["int"],
    4: left_camera_calibs["cam_4"]["int"],
    
    6: np.array(
        [
            [706.01969952, 0.0, 360.86504065],
            [0.0, 706.15628068, 490.34852859],
            [0.0, 0.0, 1.0],
        ],
    ),

    #aria
    "aria": np.array(
        [
            [610.94268799, 0.0, 703.5],
            [0.0, 610.94268799, 703.5],
            [0.0, 0.0,  1.0],
        ],
    )
    
}
D = {
    1: right_camera_calibs["cam_1"]["dist_coeff"],
    2: right_camera_calibs["cam_2"]["dist_coeff"],
    3: left_camera_calibs["cam_3"]["dist_coeff"],
    4: left_camera_calibs["cam_4"]["dist_coeff"],

    # iphone
    6: np.array(
        [
            [
                2.97673215e-01,
                -1.69844695e00,
                1.65368204e-03,
                3.61532041e-05,
                3.03597517e00,
            ],
        ]
    ),
    
    "aria": np.zeros(5, dtype=np.float32),
}

T_robot_to_camera = {
    1: right_camera_calibs["cam_1"]["ext"],
    2: right_camera_calibs["cam_2"]["ext"],
    3: left_camera_calibs["cam_3"]["ext"],
    4: left_camera_calibs["cam_4"]["ext"],
}
T_aruco_to_camera = {
    2: np.array(
        [
            [0.87833182, 0.47753213, -0.0222771, -0.00709056],
            [0.2047396, -0.41787251, -0.88513516, 0.28966185],
            [-0.43198947, 0.77288138, -0.46480047, 1.12155578],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    4: np.array(
        [
            [-0.90335721, 0.42750082, -0.03447882, 0.08824751],
            [0.21219395, 0.3756293, -0.90215096, 0.2580675],
            [-0.37271902, -0.82228078, -0.43004052, 1.20200965],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
}
