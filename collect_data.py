import hydra
from frankateach.data_collector import DataCollector


@hydra.main(config_path="configs", config_name="collect_data", version_base="1.2")
def main(cfg):
    data_collector = DataCollector(
        robot = cfg.robot,
        storage_path=cfg.storage_path,
        demo_num=cfg.demo_num,
        cams=cfg.cam_info,
        cam_config=cfg.cam_config,
        collect_img=cfg.collect_img,
        collect_depth=cfg.collect_depth,
        collect_state=cfg.collect_state,
        collect_reskin=cfg.collect_reskin,
    )
    data_collector.start()


if __name__ == "__main__":
    main()
