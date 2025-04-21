import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import json
from collections import deque
import math
import os
from env import BattleEnv

import time

from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ACTION_DIMENSION = 3




# Actor网络：输出连续动作（移动方向+射击概率）
class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Actor, self).__init__()
        # self.lstm = nn.LSTM(state_dim, state_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_DIMENSION),
            nn.Tanh() 
        )
        self.to(device)

    def forward(self, state, training=True, noise_scale=0.1):
        out = self.net(state)
        # if training:
        #     # 训练时添加噪声
        #     noise = torch.randn_like(out) * noise_scale
        #     out = torch.clamp(out + noise, -1, 1)
        return out.cpu()
    

# Critic网络：输入全局状态+所有智能体动作
class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, num_agents, hidden_dim=256):
        super(Critic, self).__init__()
        input_dim = obs_dim * num_agents + action_dim * num_agents
        self.norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            # nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )
        self.to(device)
    
    def forward(self, obss, actions):
        x = torch.cat([obss, actions], dim=1)
        x = self.norm(x)
        return self.net(x).to(device)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, actions, rewards, next_obs, done):
        self.buffer.append((obs, actions, rewards, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs = torch.FloatTensor(np.array(obs)).to(device)            # [batch, num_agents, obs_dim]
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_obs = torch.FloatTensor(np.array(next_obs)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.buffer)

class MADDPG:
    def __init__(self, env, gamma=0.99, noise=0.4, noise_decay=0.9995, epsilon=0.6, epsilon_decay=0.9995, actor_lr=1e-4, critic_lr=1e-3):
        self.env = env
        self.num_agents = env.total_agents // 2
        self.state_dim = len(env.reset())
        self.obs_dim = len(env._get_obs(0))
        self.action_dim = ACTION_DIMENSION
        
        # Initialize networks
        self.actors = [Actor(self.obs_dim, self.action_dim).to(device) for _ in range(self.num_agents)]
        self.critics = [Critic(self.obs_dim, self.action_dim, self.num_agents).to(device) for _ in range(self.num_agents)]
        self.target_actors = [Actor(self.obs_dim, self.action_dim).to(device) for _ in range(self.num_agents)]
        self.target_critics = [Critic(self.obs_dim, self.action_dim, self.num_agents).to(device) for _ in range(self.num_agents)]
        
        # Sync target networks
        for i in range(self.num_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())
        
        # Optimizers
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=critic_lr) for critic in self.critics]
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.noise_scale = noise
        self.noise_decay = noise_decay
        # self.noise_decrease = 0
        self.memory = ReplayBuffer(100000)


    def update_noise(self):
        self.noise_scale *= self.noise_decay

    def update_epsilon(self):
        if self.epsilon==0:
            return
        self.epsilon *= self.epsilon_decay

    def debug_test_critic(self, log_path, epoch_num, obs, actions):
        obs = torch.FloatTensor(np.array(obs)).view(1, -1).to(device)
        actions = torch.FloatTensor(np.array(actions)).view(1, -1).to(device)
        # print(obs.shape, actions.shape)
        with torch.no_grad():
            sample_q = self.critics[0](obs, actions)
            debug_text = f"[Debug] epoch{epoch_num} Critic[0] Q mean: {sample_q.mean():.2f}, std: {sample_q.std():.2f}\n"
        with open(log_path, "a") as lp:
            lp.write(debug_text)
    

    def select_action(self, obs_n):
        actions = []
        for i, obs in enumerate(obs_n):
            if i < self.num_agents:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                act = self.actors[i](obs_tensor)[0].detach().cpu().numpy()
                act += np.random.normal(0, self.noise_scale, size=act.shape)
                act = np.clip(act, -1, 1)
                actions.append(act)
        for i in range(len(obs_n)):
            random_action = np.random.random(3) * 2 - 1
            actions.append(random_action)
        return actions
    
    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return 0, 0

        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = self.memory.sample(batch_size)
        actor_losses, critic_losses = [], []

        for i in range(self.num_agents):
            obs = obs_batch[:, i, :]
            next_obs = next_obs_batch[:, i, :]

            all_obs = obs_batch.view(batch_size, -1)
            all_next_obs = next_obs_batch.view(batch_size, -1)
            all_actions = act_batch.view(batch_size, -1)

            with torch.no_grad():
                target_acts = [self.target_actors[j](next_obs_batch[:, j, :]) for j in range(self.num_agents)]
                target_acts_cat = torch.cat(target_acts, dim=1).to(device)
                target_q = self.target_critics[i](all_next_obs, target_acts_cat)
                target_q = rew_batch[:, i].unsqueeze(1) + (1 - done_batch) * self.gamma * target_q

            curr_q = self.critics[i](all_obs, all_actions)
            critic_loss = nn.MSELoss()(curr_q, target_q)
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            curr_acts = [self.actors[j](obs_batch[:, j, :]).detach() if j != i else self.actors[j](obs_batch[:, j, :]) for j in range(self.num_agents)]
            curr_acts_cat = torch.cat(curr_acts, dim=1).to(device)
            actor_loss = -self.critics[i](all_obs, curr_acts_cat).mean()
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

            # soft update
            for target, param in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                target.data.copy_(0.01 * param + 0.99 * target)
            for target, param in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                target.data.copy_(0.01 * param + 0.99 * target)

        if len(actor_losses)*len(critic_losses)==0:
            return 0,0
        
        return np.mean(actor_losses), np.mean(critic_losses)





def train_Half(env, actor_lr=2.5e-4, critic_lr=1e-3, episodes=3000, max_steps=200, batch_size=256, is_render=False, task_code="test", debug=False):
    
    print("Using device:", device)
    
    maddpg = MADDPG(env,
                   gamma=0.99,
                   noise=0.2,
                   noise_decay=0.9995,
                   epsilon=0.6,
                   epsilon_decay=0.999,
                   actor_lr=actor_lr,
                   critic_lr=critic_lr
                   )

    import matplotlib.pyplot as plt
    # from IPython.display import clear_output
    plt.ion()  # 开启交互模式
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 6))
    # plt.close(fig) 
    
    rewards_log = []
    episode_rewards = []
    periodical_rewards = []

    reward_history = []
    actor_loss_history = []
    critic_loss_history = []

    uniform_path = Path("uniform") / task_code
    if not os.path.exists(uniform_path):
        os.mkdir(uniform_path)

    total_step = 0

    for ep in range(episodes):

        ep_start_time = time.time()

        env.reset()
        # total_rewards = np.zeros(agent.num_agents)

        a_loss_episode = []
        c_loss_episode = []

        model_save_path = uniform_path / f"model_ep{ep}.pth"
        log_save_path = uniform_path / "log.txt"
        record_save_path = uniform_path / f"record_part_{ep//100}.jsonl"

        # obs_n = [env._get_obs(i) for i in range(env.total_agents)]
        obs_n = env._get_obs_all()[:env.total_agents // 2]
        total_rewards = np.zeros(env.total_agents // 2)

        for _ in range(max_steps):

            total_step += 1

            actions = maddpg.select_action(obs_n)
            next_obs_n, rewards, done, _ = env.step(actions, return_half_reward=True)
            next_obs_n = next_obs_n[:env.total_agents // 2]
            # next_obs_n = [env._get_obs(i) for i in range(env.total_agents)]

            if is_render:
                env.render()

            maddpg.memory.add(obs_n, actions[:env.total_agents//2], rewards, next_obs_n, float(done))
            if total_step%2 == 0:
                al, cl = maddpg.update(batch_size)

                a_loss_episode.append(al)
                c_loss_episode.append(cl)

            obs_n = next_obs_n
            total_rewards += rewards

            if done:
                break

        ep_end_time = time.time()
        time_cosumed = ep_end_time - ep_start_time

        avg_reward = total_rewards.mean()
        rewards_log.append(avg_reward)
        log_text = f"Episode {ep+1}, Reward:{avg_reward:.2f}, Noise:{maddpg.noise_scale:.3f}, epsilon:{maddpg.epsilon: .3f}, aloss:{np.mean(a_loss_episode): .3f}, closs:{np.mean(c_loss_episode): .3f}, time:{time_cosumed: .2f}"
        # print(log_text)
        with open(log_save_path, "a") as logfile:
            logfile.write(log_text+"\n")
            logfile.close()

        if ep < 1000:
            maddpg.update_noise()
            maddpg.update_epsilon()

        episode_rewards.append(total_rewards[:].mean())
        periodical_rewards.append(total_rewards[:].mean())

        env.save_and_clear(ep, record_save_path)

        reward_history.append(avg_reward)
        actor_loss_history.append(np.mean(a_loss_episode))
        critic_loss_history.append(np.mean(c_loss_episode))

        if debug:
            maddpg.debug_test_critic(log_path=uniform_path/"debug_log.txt", epoch_num=ep,
                                     obs=obs_n, actions=actions[:env.total_agents//2])
        
        # 动态更新图像
        # clear_output(wait=True)
        # plt.close(fig)
        # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 6))
        
        ax1.clear()
        ax1.plot(reward_history, label='Reward', color='blue')
        ax1.set_title(f'Episode {ep+1} - Avg Reward: {avg_reward:.2f}')
        ax1.legend()
        
        ax2.clear()
        ax2.plot(actor_loss_history, label='Actor Loss', color='red')
        ax2.set_title('Actor Loss')
        ax2.legend()
        
        ax3.clear()
        ax3.plot(critic_loss_history, label='Critic Loss', color='green')
        ax3.set_title('Critic Loss')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
        plt.pause(0.1)  # 短暂暂停避免图像闪烁
        
        if ep % 100 == 0:
            torch.save(maddpg.actors[0].state_dict(), model_save_path)
            plt.savefig(uniform_path / f"figure_ep{ep}.png")

    plt.ioff()  # 关闭交互模式
    return maddpg




# 运行训练
if __name__ == "__main__":

    # task_series = "F_commu"7
    task_code = "16_Reward_test_noise_once"

    env = BattleEnv(red_agents=2, blue_agents=2, auto_record=True)
    rewards = train_Half(env, episodes=3000, is_render=False, task_code=task_code, debug=True)

    exit(0)
    
    # 训练后测试
    state = env.reset()
    agent = MADDPG(env)
    while True:
        actions = agent.act(state)  # 关闭探索噪声
        next_state, rewards, done, _ = env.step(actions)
        env.render()
        if done:
            state = env.reset()
        else:
            state = next_state