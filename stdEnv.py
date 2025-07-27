import cv2
import numpy as np
import base64
import math
from collections.abc import Mapping
import random
import time
from contextlib import contextmanager
import gym


class GazeboMaze(gym.Env):
    '''
    '@property' is used to create a read-only property of the class.
    from the view of design pattern, it provide safer way to access the data of the class.
    some key data is only readable, like math.pi is mutable, it is a bad design
    if let me to design, I would set math.pi ad a unmutable value
    If you feel confused about the usage of '@property', you can check the official documentation:
    '''
    def __init__(
            self,
            ros_topics: Mapping,
            ros_services: Mapping,
            state_dim: tuple[int, int, int],
            robot_name: str,
            goal_name: str,
            robot_init_position: tuple[float, float],
            goal_init_position: tuple[float, float],
            scan_topic_name: str = '/scan',
            camera_topic_name: str = '/camera/rgb/image_raw',
            cmd_vel_topic_name: str = '/cmd_vel',
            ground_plane_name: str = 'ground_plane',
    ):
        self._topics = ros_topics
        self._services = ros_services
        self._img_channels = state_dim[0]
        self._img_height = state_dim[1]
        self._img_width = state_dim[2]
        self._robot_name = robot_name
        self._goal_name = goal_name
        self._robot_init_position = robot_init_position
        self._goal_init_position = goal_init_position

        self._scan_topic_name = scan_topic_name
        self._camera_topic_name = camera_topic_name
        self._cmd_vel_topic_name = cmd_vel_topic_name
        self._ground_plane_name = ground_plane_name

        self._rl_params = {  # some parameters for the RL agent
            'success': False, # whether the agent has reached the goal or not
            'done': False, # whether the episode is done or not
            'success_episode': 0, # number of episodes the agent has succeeded
            'rpd': (0.0, 0.0), # xiang dui distance and angle to the goal
            'reward': 0.0, # reward of the current episode
        }

        self._rl_constant = {  # TODO: move to construction as a **kwargs
            'r_collision': -20.0,  # reward for collision
            'r_arrive': 100.0,  # reward for arriving the goal
            'Cd': 2.0, # threshold for arriving the goal
            'Cr': 100, # yi ge xi shu
            'Cp': -0.02, # yi ge xi shu, r_notarrive = Cr * delta_d + Cp
            'a_min': 0.1,  # minimum linear velocity
            'a_max': 0.5,  # maximum linear velocity
            'w_min': -1.0,  # minimum angular velocity
            'w_max': 1.0,  # maximum angular velocity
        }

        # self._topics += self.SCAN_TOPIC_NAME
        # self._topics += self.CAMERA_TOPIC_NAME

        self.action_space = gym.spaces.Box(
            low=np.array([self._rl_constant['a_min'], self._rl_constant['w_min']]),
            high=np.array([self._rl_constant['a_max'], self._rl_constant['w_max']]),
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._img_channels, self._img_height, self._img_width),
            dtype=np.uint8
        )


    @property
    def GROUND_PLANE(self) -> str:
        return self._ground_plane_name
    
    @property
    def ROBOT_NAME(self) -> str:
        return self._robot_name
    
    @property
    def GOAL_NAME(self) -> str:
        return self._goal_name
    
    @property
    def SCAN_TOPIC_NAME(self) -> str:
        return self._scan_topic_name
    
    @property
    def CAMERA_TOPIC_NAME(self) -> str:
        return self._camera_topic_name
    
    @property
    def CMD_VEL_TOPIC_NAME(self) -> str:
        return self._cmd_vel_topic_name
    
    @property
    def is_pause(self) -> bool:
        resp = self._services['/gazebo/get_physics_properties'] << {}
        return resp['pause']

    @is_pause.setter
    def is_pause(self, value: bool):
        if value == self.is_pause:
            return
        if value:
            self._services['/gazebo/pause_physics'] << {}
        else:
            self._services['/gazebo/unpause_physics'] << {}

    @contextmanager
    def pause(self):
        orig_flag = self.is_pause
        self.is_pause = True
        try:
            yield
        finally:
            self.is_pause = orig_flag

    @contextmanager
    def unpause(self):
        orig_flag = self.is_pause
        self.is_pause = False
        try:
            yield
        finally:
            self.is_pause = orig_flag

    @property
    def rl_params(self) -> dict: return self._rl_params

    @property
    def laser_scan(self) -> np.ndarray:
        res = np.array(self._topics[self.SCAN_TOPIC_NAME]['ranges'])
        res[res == None] = 30.
        return res
    
    @property
    def laser_scan_info(self) -> dict:
        msg = self._topics[self.SCAN_TOPIC_NAME]
        return {
            'angle_min': msg['angle_min'],
            'angle_max': msg['angle_max'],
            'angle_increment': msg['angle_increment'],
            'time_increment': msg['time_increment'],
            'range_min': msg['range_min'],
            'range_max': msg['range_max'],
        }

    @property
    def raw_image(self) -> np.ndarray:
        msg = self._topics[self.CAMERA_TOPIC_NAME]
        img_data = msg['data'].encode('ascii')
        img_data = base64.b64decode(img_data)
        res = np.frombuffer(img_data, dtype=np.uint8)
        res = res.reshape(msg['height'], msg['width'], -1).astype(np.float32) / 255.0
        return cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

    @property
    def image(self) -> np.ndarray:
        img = self.raw_image
        if self._img_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self._img_width, self._img_height))
        return img.transpose(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    @property
    def cmd_vel(self) -> tuple[float, float]:
        msg = self._topics[self.CMD_VEL_TOPIC_NAME]
        return msg['linear']['x'], msg['angular']['z']
    
    @cmd_vel.setter
    def cmd_vel(self, value: tuple[float, float]):
        linear_x, angular_z = value
        self._topics[self.CMD_VEL_TOPIC_NAME]  = {
            'linear': {'x': linear_x, 'y': 0.0, 'z': 0.0},
            'angular': {'x': 0.0, 'y': 0.0, 'z': angular_z}
        }

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            self[key, self.GROUND_PLANE] = value
            return
        model_name, reference_name = key
        if model_name == 'robot':
            model_name = self.ROBOT_NAME
        if model_name == 'goal':
            model_name = self.GOAL_NAME
        if reference_name == 'robot':
            reference_name = self.ROBOT_NAME
        if reference_name == 'goal':
            reference_name = self.GOAL_NAME
        x, y, theta = value
        resp = self._services['/gazebo/set_model_state'] << {
            'model_state': {
                'model_name': model_name,
                'reference_frame': reference_name,
                'pose': {
                    'position': { 'x': x, 'y': y, 'z': 0.0},
                    'orientation': {'x': 0.0, 'y': 0.0, 'z': math.sin(theta / 2), 'w': math.cos(theta / 2)}
                }
            }       
        }
        if not resp['success']:
            raise ValueError(f"Failed to set model state: {resp['status_message']}")
        
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            return self[key, self.GROUND_PLANE]
        model_name, reference_name = key
        if model_name == 'robot':
            model_name = self.ROBOT_NAME
        if model_name == 'goal':
            model_name = self.GOAL_NAME
        if reference_name == 'robot':
            reference_name = self.ROBOT_NAME
        if reference_name == 'goal':
            reference_name = self.GOAL_NAME
        resp = self._services['/gazebo/get_model_state'] << {
           'model_name': model_name,
           'relative_entity_name': reference_name
        }
        if not resp['success']:
            raise ValueError(f"Failed to get model state: {resp['status_message']}")
        pose = resp['pose']
        return (pose['position']['x'], pose['position']['y'], 2 * math.atan2(pose['orientation']['z'], pose['orientation']['w']))
    
    def reset(self):
        self.is_pause = False
        self.cmd_vel = 0.0, 0.0
        # self._services['/gazebo/reset_simulation'] << {}
        self['robot'] = *self._robot_init_position, random.gauss(-math.pi/2, math.pi/2)
        d_x, d_y, d_theta = self['robot', 'goal']
        d_dist = math.sqrt(d_x**2 + d_y**2)
        self._rl_params['rpd'] = d_dist, d_theta
        self._rl_params['success'] = False
        self._rl_params['reward'] = 0.0
        self._rl_params['success_episode'] = 0
        self._services['/gazebo/reset_simulation'] << {}
        self['goal'] = *self._goal_init_position, 0.0
        time.sleep(3)
        self.is_pause = True
        res = self.image
        return res
    
    def step(self, action: tuple[float, float]):
        self.is_pause = False
        linear_x, angular_z = action
        self.cmd_vel = linear_x, angular_z
        time.sleep(0.1)  # TODO: move to construction
        next_state = self.image
        scan_data = self.laser_scan
        self._rl_params['done'] = False
        self._rl_params['reward'] = 0.0
        min_range = 0.2  # TODO: move to construction
        if np.min(scan_data) < min_range:
            print(f"\033[31mcollision! min_range={np.min(scan_data)}, action={action}\033[0m")
            self._rl_params['done'] = True
            self._rl_params['reward'] = self._rl_constant['r_collision']
        d_x, d_y, d_theta = self['robot', 'goal']
        d_dist = math.sqrt(d_x**2 + d_y**2)
        if d_dist < self._rl_constant['Cd']:
            print("\033[32mArrived!\033[0m")
            self._rl_params['done'] = True
            self._rl_params['success'] = True
            self._rl_params['reward'] = self._rl_constant['r_arrive']
            self._rl_params['success_episode'] += 1
        else:
            delta_d = self._rl_params['rpd'][0] - d_dist
            self._rl_params['reward'] = self._rl_constant['Cr'] * delta_d + self._rl_constant['Cp'] * (1 - math.exp(-delta_d))

        self._rl_params['rpd'] = d_dist, d_theta
        self.is_pause = True
        return next_state, self._rl_params['reward'], self._rl_params['done'], self._rl_params['success']  

    def __str__(self): return str({
        'robot_name': self._robot_name,
        'goal_name': self._goal_name,
        'robot_init_position': self._robot_init_position,
        'goal_init_position': self._goal_init_position,
        'pause': self.is_pause,
        ** self._rl_params
    })

        