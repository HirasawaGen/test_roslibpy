from utils import ros_context
import numpy as np
import gym

class ExampleAgent(gym.Env):
    def __init__(
        self,
        host: str,
        port: int,
    ):
        self._manager = ros_context(host, port)
        _, self._topics, _ = self._manager.__enter__()
        self._topics += '/cmd_vel'

    @staticmethod
    def _image2ndarray(msg: dict) -> np.ndarray:
        # convert image message to numpy array
        pass

    @property
    def cmd_vel(self) -> tuple[float, float]:
        msg = self._topics['/cmd_vel']
        linear_x = msg['linear']['x']
        angular_z = msg['angular']['z']
        return linear_x, angular_z

    @cmd_vel.setter
    def cmd_vel(self, value):
        linear_x, angular_z = value
        self._topics['/cmd_vel'] = {
            'linear': {'x': linear_x, 'y': 0, 'z': 0},
            'angular': {'x': 0, 'y': 0, 'z': angular_z}
        }

    @property
    def image(self) -> np.ndarray | None:
        linear_x, angular_z = self.cmd_vel
        if abs(linear_x) < 0.01 and abs(angular_z) < 0.01:
            return None
        msg = self._topics['/camera/image_raw']
        return self._image2ndarray(msg)

    def step(self, action):
        pass

    def reset(self):
        pass

    def close(self):
        self._manager.__exit__(None, None, None)
    