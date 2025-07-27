from tqdm import tqdm#进度条
import numpy as np
import torch


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()#列表反转
    return torch.tensor(advantage_list, dtype=torch.float)


def train_on_policy_agent(env, agent, max_episodes, max_steps):
    return_list = []
    for i in range(10):
        with tqdm(total=int(max_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(max_episodes / 10)):
                done = False  # 回合标志，重置假
                success = False
                episode_return = 0
                step = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}

                state = env.reset()

                while not done:
                    action = agent.take_action(state)  # 探索
                    next_state, reward, done, success = env.step((action[0].item(), action[1].item()))  # 行动

                    transition_dict['states'].append(state)  # 数据存储
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)

                    state = next_state  # 信息更新
                    episode_return += reward
                    step += 1

                    if success:  # 成功通知
                        print(f'Successfully Reach Goal at Episode: {max_episodes / 10 * i + i_episode + 1}')
                    if done or step == max_steps:  # 标志为True或超过最大步数，跳出此次交互
                        break

                return_list.append(episode_return)
                actor_loss_mean, critic_loss_mean = agent.update(transition_dict)

                # 单世代信息输出：
                pbar.set_postfix({'Epi': '%d' % (max_episodes / 10 * i + i_episode + 1),
                                  'Ret': '%.3f' % episode_return,
                                  'Alos': '%.3f' % actor_loss_mean,
                                  'Clos': '%.3f' % critic_loss_mean,
                                  'Suc': success})
                pbar.update(1)
    return return_list
