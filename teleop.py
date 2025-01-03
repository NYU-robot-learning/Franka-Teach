import hydra
from multiprocessing import Process
from frankateach.teleoperator import FrankaOperator
from frankateach.oculus_stick import OculusVRStickDetector
from frankateach.constants import HOST, VR_CONTROLLER_STATE_PORT


def start_teleop(init_gripper_state="open", teleop_mode="robot", home_offset=None):
    operator = FrankaOperator(
        init_gripper_state=init_gripper_state,
        teleop_mode=teleop_mode,
        home_offset=home_offset,
    )
    operator.stream()


def start_oculus_stick():
    detector = OculusVRStickDetector(HOST, VR_CONTROLLER_STATE_PORT)
    detector.stream()


@hydra.main(version_base="1.2", config_path="configs", config_name="teleop")
def main(cfg):
    teleop_process = Process(
        target=start_teleop,
        args=(
            cfg.init_gripper_state,
            cfg.teleop_mode,
            cfg.home_offset,
        ),
    )
    oculus_stick_process = Process(target=start_oculus_stick)

    teleop_process.start()
    oculus_stick_process.start()

    teleop_process.join()
    oculus_stick_process.join()


if __name__ == "__main__":
    main()
