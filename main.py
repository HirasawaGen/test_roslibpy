from utils import ros_context
from stdEnv import GazeboMaze
import math
import os
import time


# with ros_context('config.yml') as ros_ctx:
#     maze = GazeboMaze(
#         ros_ctx,
#         [3, 84, 84],
#         f'turtlebot3_{os.environ["TURTLEBOT3_MODEL"]}',
#         'unit_cylinder',
#         (1.0, 1.0),
#         (6.0, 2.0),
#     )
#     print(maze.laser_scan.shape)
#     print(maze.image.shape)
#     maze.reset()
#     maze.step((0.5, 0.0))

with ros_context('localhost', 9090) as (client, topics, services):
    maze = GazeboMaze(
        topics,
        services,
        [3, 84, 84],
        # f'turtlebot3_{os.environ["TURTLEBOT3_MODEL"]}',
        f'fetch',
        'table1',
        (0.0, 0.0),
        (6.0, 2.0),
        scan_topic_name='/base_scan',
        camera_topic_name='/head_camera/rgb/image_raw',
    )
    print(maze['robot'])
    print(maze.laser_scan.shape)
    print(maze.image.shape)
    start_time = time.time()
    while time.time() - start_time < 3.0:
        res = maze.step((0.2, 0.0))
        print(res)
    maze.reset()
    