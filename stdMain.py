"""
此脚本是本工程强化学习方面的主函数脚本：
    1. 关于如何使用argparse:
        运行如下指令：python3 stdMain --ParaXXX 2000
"""
import argparse
from stdAgent import PPOContinuous
from stdEnv import GazeboMaze
from stdAgent import CNN_Model
from stdUtils import train_on_policy_agent, moving_average
import matplotlib.pyplot as plt
import matplotlib
import torch
from utils import ros_context
import time
import os


if __name__ == '__main__':

    """ 
    @第一步：参数配置：
        基于配置文件stdConfig.py，并结合ArgumentParser，实现动态调参
    """
    parser = argparse.ArgumentParser()  # 参数解析
    parser.add_argument('--ParaXXX', type=int, default=1000)  # 迭代的世代次数
    args = parser.parse_args()  # 解析命令行变量，传递给args，字典形式
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    """
    @第二步：角色设置：
        Agent: PPOContinuous()
        Environment: GazeboMaze()
    """
    agent = PPOContinuous(3, 128, 2, 1, 1e-4, 1e-3, 0.9, 200,
                          0.2, 0.9, device)
    with ros_context('localhost', 9090) as (client, topics, services):
        env = GazeboMaze(
            topics,
            services,
            [3, 84, 84],
            # f'turtlebot3_{os.environ["TURTLEBOT3_MODEL"]}',
            f'fetch',
            'unit_cylinder',
            (1.0, 1.0),
            (6.0, 2.0),
            scan_topic_name='/base_scan',
            camera_topic_name='/head_camera/rgb/image_raw',
            # cmd_vel_topic_name='/teleop/cmd_vel',
        )
        env.reset()
        env.is_pause = False
        print(env.image.shape)
        print(env.laser_scan.shape)
        # exit()
        # while time.time() - start_time < 10:
        #     env.cmd_vel = 0.1, 0.0
        #     time.sleep(0.1)
        #     print('publishing cmd_vel: 0.1, 0.0')
            
        # exit()
        topics += env.SCAN_TOPIC_NAME
        # topics += env.CAMERA_TOPIC_NAME

        """ 
        @第三步：循环训练: 
        """
        return_list = train_on_policy_agent(env, agent, 1000, 512)
        print('Training finished.')

    os.remove('./output_1.png')
    os.remove('./output_2.png')

    """ @最后一步：画图"""
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format('fetch'))
    # plt.show()
    plt.savefig("./output_1.png")

    mv_return = moving_average(return_list, 21)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format('fetch'))
    # plt.show()
    plt.savefig("./output_2.png")