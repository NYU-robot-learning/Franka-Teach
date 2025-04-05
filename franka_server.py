# import argparse
# import os

# parser = argparse.ArgumentParser()
# parser.add_argument("--hand", type=str, default="right", choices=["left", "right"],
#                     help="Specify which hand to use: left or right")
# args, _ = parser.parse_known_args()
# os.environ["HAND"] = args.hand

# from frankateach.franka_server import FrankaServer
# import hydra

# @hydra.main(version_base="1.2", config_path="configs", config_name="franka_server")
# def main(cfg):
#     fs = FrankaServer(cfg.deoxys_config_path)
#     # fs = FrankaServer(cfg.deoxys_config_path, control_port=cfg.control_port)
#     fs.init_server()


# if __name__ == "__main__":
#     main()

import sys
import os

# Default hand is "right"
hand = "right"
# Look for "--hand=" in sys.argv, and if found, extract its value and remove it.
for arg in sys.argv:
    if arg.startswith("--hand="):
        hand = arg.split("=")[1]
        sys.argv.remove(arg)
        break

# Set an environment variable (or a global variable) based on the hand value.
os.environ["HAND"] = hand

# Now, when Hydra parses sys.argv, it won't see "--hand"
import hydra
from frankateach.franka_server import FrankaServer

@hydra.main(version_base="1.2", config_path="configs", config_name="franka_server")
def main(cfg):
    # Override the deoxys_config_path based on the global hand info.
    if os.environ.get("HAND") == "left":
        cfg.deoxys_config_path = "deoxys_left.yml"
    else:
        cfg.deoxys_config_path = "deoxys_right.yml"

    fs = FrankaServer(cfg.deoxys_config_path)
    fs.init_server()

if __name__ == "__main__":
    main()

