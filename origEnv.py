"""
此脚本是本工程强化学习方面的环境函数：
"""
import rospy
import tf
import copy
import numpy as np
import time
import math
from gazebo_msgs.srv import GetModelState  # 引入消息包
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import Image  # 传感器数据类型
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, Point, Quaternion, Twist  # 速度(线速度与角速度)数据类型
from std_srvs.srv import Empty  # 空操作数据类型
import cv2  # 图像处理包，导入OpenCV库
from cv_bridge import CvBridge, CvBridgeError  # v_bridge，用于图像处理和在OpenCV和ROS图像消息之间转换
from stdConfig import *  # 参数


class GazeboMaze:
    """
    Gazebo的环境类实现：
        1. __init__方法：创建ROS节点，并初始化相关参数
        2. reset方法，Gazebo环境重置：
            (1) Resets the state of the environment
            (2) Returns an initial observation
        3. step方法，机器人执行指令，观测新状态：
            (1) Execute action
            (2) Observes next state, reward, and done
    """
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('GazeboMaze')  # 初始化ROS节点，此节点的名字是GazeboMaze
        time.sleep(10)  # 等待gazebo服务器启动

        # 目标位置
        self.goal = []
        self.start = []

        # 建立通信
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)  # 获取和设置模型状态，
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)  # 最后一个参数是数据格式

        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)  # 重置仿真环境
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)  # 物理引擎的恢复和暂停
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)  # 创建发布者，向相关Topic发布信息

        self.laser_sub = rospy.Subscriber('scan', LaserScan, self.LaserScanCallBack)
        # 订阅激光扫描数据话题scan，并通过回调函数LaserScanCallBack处理该话题数据，有数据就调用LaserScanCallBack处理

        # 数据形状: 状态数据形状
        self.img_channels = state_dim[0]  # 输入数据的size
        self.img_height = state_dim[1]
        self.img_width = state_dim[2]

        # 机器人速度指令初始化
        self.vel_cmd = [0., 0.]

        # 奖励、成功标志等设置
        self.reward = 0  # 用于存储奖励值
        self.success = False  # 表示是否成功达成目标
        self.success_episodes = 0  # 用于追踪机器人成功完成任务的次数

        # 其余属性
        self.success = None  # 成功探索标志，在step()方法中的done是完成探索标志；done包含碰撞和到目标点
        self.reward = None
        self.vel_cmd = None
        self.rpd = None  # 相对的位置和朝向
        self.scan = None
        self.scan_param = None

    def LaserScanCallBack(self, scan):
        """
        回调函数LaserScanCallBack，处理该scan话题数据；
        self.scan_param: 将scan消息中的一些关键参数提取出来，并保存到self.scan_param中;
        self.scan: 将scan中的ranges数据转化成数组;
        """
        self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment, scan.scan_time,
                           scan.range_min, scan.range_max]
        self.scan = np.array(scan.ranges)

    def set_goal(self, x, y):
        """ 设置目标的位置 """
        state = SetModelState()
        # 设置模型的名称和参考系
        state.model_name = 'unit_cylinder'
        state.reference_frame = 'world'  # ''ground_plane'
        # 设置模型的位姿
        state.pose = Pose()
        state.pose.position = Point(x, y, 0.135813)
        state.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        # 设置模型的速度
        state.twist = Twist()
        state.twist.linear = Point(0.0, 0.0, 0.0)
        state.twist.angular = Point(0.0, 0.0, 0.0)

        # 调用服务，设置位置
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            result = self.set_state(state)
            assert result.success is True
        except rospy.ServiceException:
            print("/gazebo/get_model_state service call failed")

    def set_start(self, x, y, theta):
        """ 设置机器人的初始位置 """
        state = SetModelState()
        # 设置模型的名称和参考系
        state.model_name = 'turtlebot3_waffle_pi'
        state.reference_frame = 'world'  # ''ground_plane'
        # 设置模型的位姿
        state.pose = Pose()
        state.pose.position = Point(x, y, 0.0)
        quaternion = tf.transformations.quaternion_from_euler(0, 0, theta)
        state.pose.orientation = Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
        # 设置模型的速度
        state.twist = Twist()
        state.twist.linear = Point(0.0, 0.0, 0.0)
        state.twist.angular = Point(0.0, 0.0, 0.0)

        # 调用服务，设置位置
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            result = self.set_state(state)
            assert result.success is True
        except rospy.ServiceException:
            print("/gazebo/get_model_state service call failed")

    def goal2robot(self, d_x, d_y, theta):
        """ goal2robot方法计算目标相对于机器人位置的距离d和角度aplha """
        pp = math.sqrt(d_x * d_x + d_y * d_y)  # 距离
        dd = math.atan2(d_y, d_x) - theta  # 角度差
        return pp, dd

    def reset(self):
        # 重置目标点位置
        self.goal = [6, 2]
        self.set_goal(self.goal[0], self.goal[1])  # 调用self.set_goal()方法，设置目标

        # 重置机器人初始位置
        self.start = [1, 1]  # 起始位置是原点，但是起始航向角度随机
        theta = np.random.uniform(- 1.0 / 2.0 * math.pi, 1.0 / 2.0 * math.pi)  # 1.0/2*math.pi  # 4.0/3*math.pi  #
        self.set_start(self.start[0], self.start[1], theta)

        # 计算目标和机器人的相对位置和朝向
        p0, d0 = self.goal2robot(self.goal[0] - self.start[0], self.goal[1] - self.start[1], theta)
        self.rpd = [p0, d0]

        # 重置标志、奖励和机器人速度命令
        self.success = False  # 重置标志为False，表示此时交互任务尚未完成
        self.reward = 0  # 重置奖励为0
        self.vel_cmd = [0., 0.]  # 重置机器人的速度命令为零

        # 重置整个环境，并开始仿真
        rospy.wait_for_service('/gazebo/reset_simulation')  # 重置整个环境
        try:
            self.reset_proxy
        except rospy.ServiceException:
            print("/gazebo/reset_simulation service call failed")

        rospy.wait_for_service('/gazebo/unpause_physics')  # 开始Gazebo仿真
        try:
            self.unpause
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")

        # 返回环境观测状态：获取图像数据并转换为OpenCV图像格式
        image_data = None
        cv_image = None
        while image_data is None:
            image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image)
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")#8位无符号整数的BGR颜色模式

        rospy.wait_for_service('/gazebo/pause_physics')  # 暂停仿真，避免在收集数据时干扰
        try:
            self.pause
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")

        if self.img_channels == 1:  # 图像处理
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            # 如果图像是单通道（灰度图），将其转换为灰度图像
        cv_image = cv2.resize(cv_image, (self.img_width, self.img_height))  # width, height
        cv_image_transposed = cv_image.transpose(2, 0, 1)
        state = cv_image_transposed / 255

        return state

    def GetLaserObservation(self):
        """
        获取激光雷达扫描数据，并对其中的无效值进行处理;
        copy.deepcopy(self.scan)，使用deepcopy方法创建self.scan的深拷贝，这样不会引用原始数据，好处是：更改scan数据而不影响self.scan；
        scan[np.isnan(scan)] = 30.，处理数据中的NaN值，替换为30.0;
        """
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(scan)] = 30.#NaN非数字元素替换为30
        return scan

    def step(self, action):
        # 解除Gazebo仿真暂停
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")

        # 根据action指令，驱动机器人(带有速度幅度限制)
        vel_cmd = Twist()
        vel_cmd.linear.x = np.clip(action[0], a_min, a_max)  # clip裁剪，根据上下限
        vel_cmd.angular.z = np.clip(action[1], w_min, w_max)
        self.pub_cmd_vel.publish(vel_cmd)  # turtlebot3, 将速度命令发布到/cmd_vel主题（控制机器人）
        time.sleep(0.05)

        # 观测新状态
        image_data = None
        cv_image = None
        while image_data is None:
            image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
        if self.img_channels == 1:  # 图像处理
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            # 如果图像是单通道（灰度图），将其转换为灰度图像
        cv_image = cv2.resize(cv_image, (self.img_width, self.img_height))  # width, height
        cv_image_transposed = cv_image.transpose(2, 0, 1)
        next_state = cv_image_transposed / 255

        # 重置：探索完成标志和回报置零
        done = False  # 探索完成标志(发生碰撞或到达目的地均视为探索完成)
        self.reward = 0

        # 碰撞判断
        scan_data = None
        while scan_data is None:
            try:
                scan_data = rospy.wait_for_message('scan', LaserScan, timeout=10)
            except:
                pass
        min_range = 0.15  # 碰撞判断
        laser = self.GetLaserObservation()
        if np.amin(laser) < min_range:
            done = True
            self.reward = r_collision
            print("\033[31mcollision!\033[0m")

        # 获取机器人位置和方向
        robot_state = None
        rospy.wait_for_service('/gazebo/get_model_state')  # 等待服务
        try:
            robot_state = self.get_state("turtlebot3_waffle_pi", "world")  # 创建服务代理API接口
            assert robot_state.success is True
        except rospy.ServiceException:
            print("/gazebo/get_model_state service call failed")

        # 计算距离目标的相对位置和方向，并进行相关判断
        position = robot_state.pose.position
        orientation = robot_state.pose.orientation
        d_x = self.goal[0] - position.x  # 计算目标点相对于机器人的位置偏差
        d_y = self.goal[1] - position.y
        _, _, theta = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z,
                                                                orientation.w])  # 将四元数转换为欧拉角（theta）
        ppp, ddd = self.goal2robot(d_x, d_y, theta)  # 调用goal2robot方法计算目标相对于机器人位置的距离d和角度aplha

        # 根据计算的相对位置和方向进行判断
        """
        如果机器人距离目标ppp小于设定阈值Cd,则认为达到目标，设置done=True,奖励为r_arrive;
        """
        if ppp < Cd:  # 到达目标
            done = True
            self.reward = r_arrive
            self.success = True
            self.success_episodes += 1
        if not done:  # 未到达目标
            delta_d = self.rpd[0] - ppp
            self.reward = Cr * delta_d + Cp

        # 更新相对位置
        self.rpd = [ppp, ddd]

        # 暂停仿真：调用ROS服务暂停仿真，防止仿真继续运行，确保当前时间步的状态已收集完毕
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")

        return next_state, self.reward, done, self.success
