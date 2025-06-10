from collections import deque
import time
from frankateach.constants import HOST
from frankateach.network import ZMQKeypointPublisher, ZMQKeypointSubscriber
from frankateach.utils import FrequencyTimer, notify_component_start

from reskin_sensor import ReSkinProcess


class ReskinSensorPublisher:
    def __init__(self, reskin_config, port):
        self.reskin_publisher = ZMQKeypointPublisher(HOST, port)

        self.timer = FrequencyTimer(100)
        self.reskin_config = reskin_config
        if reskin_config.history is not None:
            self.history = deque(maxlen=reskin_config.history)
        else:
            self.history = deque(maxlen=1)
        self._start_reskin()

    def _start_reskin(self):
        self.sensor_proc = ReSkinProcess(
            num_mags=self.reskin_config["num_mags"],
            port=self.reskin_config["port"],
            baudrate=100000,
            burst_mode=True,
            device_id=0,
            temp_filtered=True,
            reskin_data_struct=True,
        )
        self.sensor_proc.start()
        time.sleep(0.5)

    def stream(self):
        notify_component_start("Reskin sensors")

        while True:
            try:
                self.timer.start_loop()
                reskin_state = self.sensor_proc.get_data(1)[0]
                data_dict = {}
                data_dict["timestamp"] = reskin_state.time
                data_dict["sensor_values"] = reskin_state.data
                self.history.append(reskin_state.data)
                data_dict["sensor_history"] = list(self.history)
                self.reskin_publisher.pub_keypoints(data_dict, topic_name="reskin")
                self.timer.end_loop()

            except KeyboardInterrupt:
                break


class ReskinSensorSubscriber:
    def __init__(self, port):
        self.port = port
        self.reskin_subscriber = ZMQKeypointSubscriber(HOST, port, topic="reskin")

    def __repr__(self):
        return "reskin"

    def get_sensor_state(self):
        reskin_state = self.reskin_subscriber.recv_keypoints()
        return reskin_state
