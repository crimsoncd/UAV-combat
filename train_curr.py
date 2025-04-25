# train_curriculum_maddpg.py
# 基于课程学习、共享Actor、改进动作采样的 MADDPG 训练主文件

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from pathlib import Path
from collections import deque
import time
import json
from env import BattleEnv

# ============ 网络结构定义 ============
class SharedActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super(SharedActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.sigma_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = self.net(obs)
        mu = torch.tanh(self.mu_head(x))
        sigma = torch.clamp(self.sigma_head(x), min=-2, max=1)
        return mu, sigma

    def sample_action(self, obs):
        mu, sigma = self.forward(obs)
        dist = torch.distributions.Normal(mu, sigma.exp())
        action = dist.sample()
        return torch.clamp(action, -1.0, 1.0), dist

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

# ============ Replay Buffer ============
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

# ============ 奖励函数（示意） ============
def simple_reward_scheme(env, task_type="task1"):
    rewards = np.zeros(env.total_agents)
    for i, drone in enumerate(env.drones):
        if not drone.alive:
            rewards[i] -= 1
        for enemy in env.get_opponents(i):
            if not enemy.alive:
                if task_type == "task2" and enemy.id == 0:
                    rewards[i] += 3
                else:
                    rewards[i] += 1
    return rewards

# ============ 主训练函数 ============
def train_curriculum(env, actor_lr=1e-4, critic_lr=1e-3, episodes=3000, batch_size=256, task_code="TaskCurr", is_render=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_agents = env.red_num
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

    noise_scale = 0.4
    noise_decay = 0.995

    save_dir = Path("curriculum_logs") / task_code
    save_dir.mkdir(exist_ok=True, parents=True)

    current_task = "task1"
    switch_episode = 1000

    for ep in range(episodes):
        env.reset()
        obs_n = env._get_obs_all()[:num_agents]
        episode_reward = 0

        for step in range(200):
            obs_tensor = torch.FloatTensor(np.array(obs_n)).to(device)
            actions, _ = actor.sample_action(obs_tensor)
            actions = actions.cpu().detach().numpy()

            # 蓝方动作为随机
            for _ in range(env.blue_num):
                actions = np.vstack([actions, np.random.uniform(-1, 1, size=action_dim)])

            next_obs_n, _, done, _ = env.step(actions)
            next_obs_n = next_obs_n[:num_agents]
            rewards = simple_reward_scheme(env, current_task)[:num_agents]

            buffer.add(obs_n, actions[:num_agents], rewards, next_obs_n, float(done))
            episode_reward += np.mean(rewards)
            obs_n = next_obs_n

            if done:
                break

        if ep > 100 and len(buffer) > batch_size:
            obs_b, act_b, rew_b, next_obs_b, done_b = buffer.sample(batch_size)
            obs_all = obs_b.view(batch_size, -1).to(device)
            next_obs_all = next_obs_b.view(batch_size, -1).to(device)
            act_all = act_b.view(batch_size, -1).to(device)

            with torch.no_grad():
                target_acts, _ = target_actor.sample_action(next_obs_b.to(device))
                target_acts_all = target_acts.view(batch_size, -1)
                q_target = rew_b.sum(1, keepdim=True).to(device) + 0.95 * target_critic(next_obs_all, target_acts_all)

            q_val = critic(obs_all, act_all)
            critic_loss = nn.MSELoss()(q_val, q_target)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            new_acts, _ = actor.sample_action(obs_b.to(device))
            actor_loss = -critic(obs_all, new_acts.view(batch_size, -1)).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            # Soft update
            for tp, p in zip(target_actor.parameters(), actor.parameters()):
                tp.data.copy_(0.01 * p.data + 0.99 * tp.data)
            for tp, p in zip(target_critic.parameters(), critic.parameters()):
                tp.data.copy_(0.01 * p.data + 0.99 * tp.data)

        print(f"[Ep {ep}] Reward: {episode_reward:.2f}, Task: {current_task}")

        if ep == switch_episode:
            current_task = "task2"

        if ep % 100 == 0:
            torch.save(actor.state_dict(), save_dir / f"actor_ep{ep}.pth")
            torch.save(critic.state_dict(), save_dir / f"critic_ep{ep}.pth")

    return actor

if __name__ == "__main__":
    env = BattleEnv(red_agents=3, blue_agents=3)
    train_curriculum(env)
