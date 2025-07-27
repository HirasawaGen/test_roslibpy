"""
此脚本是本工程强化学习方面的Agent脚本：
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import stdUtils


class ChannelAttention(nn.Module):
    """ 通道注意力模块 """
    def __init__(self, in_planes, ratio=16):#输入特征通道数，缩放比例
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)#通道注意力权重


class SpatialAttention(nn.Module):
    """ 空间注意力模块 """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)#—忽略返回的第二个值，索引
        x = torch.cat([avg_out, max_out], dim=1)#拼接
        x = self.conv1(x)
        return self.sigmoid(x)


class CNN_Model(nn.Module):
    """ CNN based backbone model """
    def __init__(self, input_channels, output_dim):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, stride=2, padding=1)#步长2
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        """
        Parameter calculate:
        (I - K + 2*P) / S + 1
        where: I : image size, the initial image size is 84x84, so I == 84 here.
               K : kernel size, here is 3
               P : padding size, here is 1
               S : stride, here is 2
        """
        self.ca1 = ChannelAttention(32)
        self.sa1 = SpatialAttention()
        self.fc1 = nn.Linear(32 * 6 * 6, 512)
        self.fc_out = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        x = x.view(x.size(0), -1)#形状变换，（批量大小，自动计算其他维度大小）展成一维向量
        x = F.relu(self.fc1(x))
        out = self.fc_out(x)
        return out.squeeze()#去除维度为1的维度


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_channels, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.backbone = CNN_Model(state_channels, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)#动作分布的均值
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)#标准差
        self.action_bound = action_bound

    def forward(self, x):
        x = self.backbone(x)
        mu_vel = self.fc_mu(x)
        if mu_vel.dim() == 1:
            mu_linear = mu_vel[0]
            mu_angular = mu_vel[1]
            mu_linear = 0.2 * torch.tanh(mu_linear) + 0.3  # 映射到[0.1, 0.5]
            mu_angular = self.action_bound * torch.tanh(mu_angular)
            mu = torch.stack([mu_linear, mu_angular], dim=0)#行方向堆叠成新张量
        else:
            mu_linear = mu_vel[:, 0]#所有
            mu_angular = mu_vel[:, 1]
            mu_linear = 0.2 * torch.tanh(mu_linear) + 0.3  # 映射到[0.1, 0.5]
            mu_angular = self.action_bound * torch.tanh(mu_angular)
            mu = torch.stack([mu_linear, mu_angular], dim=1)#列方向
        std = F.softplus(self.fc_std(x))#对数，F映射到正值
        std = torch.clamp(std, min=1e-6)  # 保证最小值大于1e-6，防止出现Normal(mu, std)参数违法
        return mu, std


class ValueNet(torch.nn.Module):
    def __init__(self, state_channels):
        super(ValueNet, self).__init__()
        self.net = CNN_Model(state_channels, 1)

    def forward(self, x):
        x = self.net(x)
        x = torch.squeeze(x, -1)
        return x


class PPOContinuous:
    ''' 处理连续动作的PPO算法 '''
    def __init__(self, state_channels, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, lmbda, max_epochs,
                 eps, gamma, device):
        self.actor = PolicyNetContinuous(state_channels, hidden_dim, action_dim, action_bound).to(device)
        self.critic = ValueNet(state_channels).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma#类
        self.lmbda = lmbda
        self.max_epochs = max_epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)
        mu, sigma = self.actor(state)#均值，标准差
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor([item.detach().cpu().numpy() for item in transition_dict['actions']]).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = stdUtils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(actions)  # 动作是正态分布

        # 训练损失
        actor_loss_sum = []
        critic_loss_sum = []
        for _ in range(self.max_epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)#新旧策略动作概率的比值
            surr1 = ratio * advantage.view(-1, 1)
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage.view(-1, 1)
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            actor_loss_sum.append(actor_loss.data.cpu().numpy())
            critic_loss_sum.append(critic_loss.data.cpu().numpy())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()#计算梯度
            critic_loss.backward()
            self.actor_optimizer.step()#更新梯度参数
            self.critic_optimizer.step()

        return np.mean(actor_loss_sum), np.mean(critic_loss_sum)
