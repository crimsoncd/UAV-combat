# 含有步骤引导的e-greedy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
from pathlib import Path
from collections import deque
import time
import json
import matplotlib.pyplot as plt
from env import BattleEnv

class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()
    
    def reset(self):
        """重置噪声状态"""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        """生成噪声样本"""
        dx = self.theta * (self.mu - self.state) 
        dx += self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

class SharedActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super(SharedActor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_dim)
        # self.ou_noise = OUNoise(action_dim)  # 添加OU噪声
        self.init_weights()
    
    def init_weights(self):
        # 使用Xavier初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)
    
    def forward(self, x):

        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = torch.tanh(self.fc4(x))
        return action
    
    def sample_action(self, x, noise=0):
        # if noise > 0:
        #     out = self.forward(x)
        #     noise_add = self.ou_noise.sample()
        #     out = torch.clamp(out + noise_add, -1, 1)
        #     return out
        return self.forward(x)

class Critic(nn.Module):
    def __init__(self, global_obs_dim, action_dim, num_agents, hidden_dim=256):
        super(Critic, self).__init__()
        input_dim = global_obs_dim + action_dim * num_agents
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs_all, acts_all):
        return self.net(torch.cat([obs_all, acts_all], dim=-1))

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, acts, rewards, next_obs, done):
        self.buffer.append((obs, acts, rewards, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(lambda x: torch.tensor(np.array(x), dtype=torch.float32), zip(*batch))

    def __len__(self):
        return len(self.buffer)

def expert_learn(ep_num):
    use_induce_prob = min(1, ep_num/1000)
    return np.random.random() < use_induce_prob

def train_curriculum(env, actor_lr=1e-4, critic_lr=1e-3, noise_scale=0.2, noise_decay=0.99,
                     episodes=3000, batch_size=256, task_code="TaskCurr", is_render=False, dev_render_trail=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_agents = env.red_agents
    obs_dim = len(env._get_obs(0))
    global_obs_dim = obs_dim * num_agents
    action_dim = 3

    actor = SharedActor(obs_dim, action_dim).to(device)
    critic = Critic(global_obs_dim, action_dim, num_agents).to(device)
    target_actor = SharedActor(obs_dim, action_dim).to(device)
    target_critic = Critic(global_obs_dim, action_dim, num_agents).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)

    buffer = ReplayBuffer()
    noise_scale = noise_scale
    noise_decay = noise_decay
    ou_noise = OUNoise(action_dim, sigma=noise_scale)

    save_dir = Path("uniform") / task_code
    save_dir.mkdir(exist_ok=True, parents=True)

    reward_history = []
    reward_enemy_history = []
    actor_loss_history = []
    critic_loss_history = []

    red_win_ep = 0
    blue_win_ep = 0
    red_win_rate = []
    blue_win_rate = []

    fig, axs = plt.subplots(2, 2, figsize=(10, 4))
    ax1, ax2, ax3, ax4 = axs[0,0], axs[0,1], axs[1,0], axs[1,1]

    current_task = "task1"

    for ep in range(episodes + 1):
        ep_start_time = time.time()
        env.reset()
        ou_noise.reset()

        obs_n = env._get_obs_all()[:num_agents]
        episode_reward = 0
        episode_reward_enemy = 0
        a_loss_episode, c_loss_episode = 0, 0

        for step in range(200):

            actions = []

            # obs_tensor = torch.FloatTensor(np.array(obs_n)).to(device)
            # actions, _ = actor.sample_action(obs_tensor)
            # actions = actions.cpu().detach().numpy()

            for r in range(env.red_agents):
                if expert_learn(ep):
                    actions.append(env.induce_step(r))
                else:
                    obs_tensor = torch.FloatTensor(np.array(obs_n[r])).to(device)
                    action = actor.sample_action(obs_tensor, noise=noise_scale).cpu().detach().numpy()
                    noise = ou_noise.sample()
                    action = np.clip(action + noise, -1, 1)
                    actions.append(action)

            for b in range(env.blue_agents):
                # actions = np.vstack([actions, np.random.uniform(-1, 1, size=action_dim)])
                # actions = np.vstack([actions, np.zeros(3)])
                actions.append([0, 0, 0])

            # print(actions)

            next_obs_n, rewards, done, _ = env.step(actions, reward_type=current_task)
            next_obs_n = next_obs_n[:num_agents]
            rewards_enemy = rewards[num_agents:]
            rewards = rewards[:num_agents]
            # rewards = simple_reward_scheme(env, current_task)[:num_agents]

            buffer.add(obs_n, actions[:num_agents], rewards, next_obs_n, float(done))
            episode_reward += np.mean(rewards)
            episode_reward_enemy += np.mean(rewards_enemy)
            obs_n = next_obs_n

            if is_render:
                env.render(show_trail=dev_render_trail)

            if done:
                break

        if ep > 100 and len(buffer) > batch_size:
            obs_b, act_b, rew_b, next_obs_b, done_b = buffer.sample(batch_size)
            obs_all = obs_b.view(batch_size, -1).to(device)
            next_obs_all = next_obs_b.view(batch_size, -1).to(device)
            act_all = act_b.view(batch_size, -1).to(device)

            with torch.no_grad():
                target_acts = target_actor.sample_action(next_obs_b.to(device))
                target_acts_all = target_acts.view(batch_size, -1)
                q_target = rew_b.sum(1, keepdim=True).to(device) + 0.95 * target_critic(next_obs_all, target_acts_all)

            q_val = critic(obs_all, act_all)
            critic_loss = nn.MSELoss()(q_val, q_target)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            new_acts = actor.sample_action(obs_b.to(device))
            actor_loss = -critic(obs_all, new_acts.view(batch_size, -1)).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            for tp, p in zip(target_actor.parameters(), actor.parameters()):
                tp.data.copy_(0.01 * p.data + 0.99 * tp.data)
            for tp, p in zip(target_critic.parameters(), critic.parameters()):
                tp.data.copy_(0.01 * p.data + 0.99 * tp.data)

            actor_loss_history.append(actor_loss.item())
            critic_loss_history.append(critic_loss.item())

            a_loss_episode = actor_loss.item()
            c_loss_episode = critic_loss.item()

        ep_end_time = time.time()
        reward_history.append(episode_reward)
        reward_enemy_history.append(episode_reward_enemy)

        outcome = env.decide_outcome()
        if 'red' in outcome:
            red_win_ep += 1
        elif 'blue' in outcome:
            blue_win_ep += 1
        red_win_rate.append(red_win_ep / (ep+1))
        blue_win_rate.append(blue_win_ep / (ep+1))

        # noise_scale *= noise_decay
        if ep%10==0:
            noise_scale *= noise_decay

        log_text = (f"[Ep {ep}] Reward: {episode_reward:.2f}, Task: {current_task}, a_loss: {a_loss_episode:.2f}, c_loss:{c_loss_episode:.2f}, "
                   f"Noise: {noise_scale:.2f}, Time: {ep_end_time - ep_start_time:.2f}, Outcome: {outcome}")
        print(log_text)
        with open(save_dir / "log.txt", "a") as f:
            f.write(log_text + "\n")

        env.save_and_clear(ep, save_dir / f"record_part_{ep//100}.jsonl")
        env.save_and_clear_rewards(ep, save_dir / f"reward_part_{ep//100}.csv")

        ax1.clear()
        ax1.plot(reward_history, label='Red Reward', color='red')
        ax1.plot(reward_enemy_history, label='Blue Reward', color='blue')
        ax1.set_title("Average Reward")
        ax1.legend()

        ax2.clear()
        ax2.plot(red_win_rate, label='Red Win Rate', color='red')
        ax2.plot(blue_win_rate, label='Blue Win Rate', color='blue')
        ax2.set_title('Win rate')
        ax2.legend()

        ax3.clear()
        ax3.plot(actor_loss_history, label='Actor Loss', color='red')
        ax3.set_title('Actor Loss')
        ax3.legend()

        ax4.clear()
        ax4.plot(critic_loss_history, label='Critic Loss', color='green')
        ax4.set_title('Critic Loss')
        ax4.legend()

        plt.tight_layout()
        if ep % 100 == 0:
            # torch.save(actor.state_dict(), save_dir / f"actor_ep{ep}.pth")
            # torch.save(critic.state_dict(), save_dir / f"critic_ep{ep}.pth")
            plt.savefig(save_dir / f"figure_ep{ep}.png")

    return actor



if __name__ == "__main__":

    # task_series = "F_commu"7
    task_code = "23_Mix_Expert_MADDPG_AZ_c"

    env = BattleEnv(red_agents=3,
                    blue_agents=3,
                    auto_record=True,
                    developer_tools=True,
                    margin_crash=False,
                    collision_crash=False)
    rewards = train_curriculum(env,
                               episodes=3000,
                               noise_scale=0,
                               noise_decay=0.99,
                               task_code=task_code,
                               is_render=False,
                               dev_render_trail=True)