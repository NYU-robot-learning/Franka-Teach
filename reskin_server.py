import hydra
from frankateach.constants import PORTS
from frankateach.sensors.reskin import ReskinSensorPublisher


@hydra.main(version_base="1.2", config_path="configs", config_name="reskin")
def main(cfg):
    port = PORTS[cfg.robot]["reskin"]
    reskin_publisher = ReskinSensorPublisher(reskin_config=cfg.reskin_config, port=port)
    reskin_publisher.stream()


if __name__ == "__main__":
    main()
