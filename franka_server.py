import sys
import hydra
from frankateach.franka_server import FrankaServer
from frankateach.constants import CONTROL_PORT_RIGHT, CONTROL_PORT_LEFT

# Default hand is "right"
side = "right"
# Look for "--side=" in sys.argv, and if found, extract its value and remove it.
for arg in sys.argv:
    if arg.startswith("--side="):
        side = arg.split("=")[1]
        sys.argv.remove(arg)
        break
config_name = f"franka_server_{side}"


@hydra.main(version_base="1.2", config_path="configs", config_name=config_name)
def main(cfg):
    control_port = CONTROL_PORT_RIGHT if side == "right" else CONTROL_PORT_LEFT
    fs = FrankaServer(cfg.deoxys_config_path, control_port)
    fs.init_server()


if __name__ == "__main__":
    main()
