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

from env_utils import noise_mask
import time

from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ACTION_DIMENSION = 3


# class OUNoise:
#     def __init__(self, action_dim=ACTION_DIMENSION, mu=0.0, theta=0.2, sigma=0.2, scale=0.3):
#         self.action_dim = action_dim
#         self.mu = mu          # 均值
#         self.theta = theta    # 回归速度
#         self.sigma = sigma    # 扩散系数
#         self.scale = scale    # 初始噪声强度
#         self.state = np.ones(action_dim) * self.mu

#     def reset(self):
#         self.state = np.ones(self.action_dim) * self.mu

#     def sample(self):
#         dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
#         self.state += dx
#         return self.state * self.scale




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
        if training:
            # 训练时添加噪声
            noise = torch.randn_like(out) * noise_scale
            out = torch.clamp(out + noise, -1, 1)
        return out.cpu()
    

# Critic网络：输入全局状态+所有智能体动作
class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, num_agents, hidden_dim=256):
        super(Critic, self).__init__()
        input_dim = obs_dim * num_agents + action_dim * num_agents
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )
        self.to(device)
    
    def forward(self, obss, actions):
        x = torch.cat([obss, actions], dim=1)
        return self.net(x).to(device)

class ReplayBuffer(deque):
    def __init__(self, capacity):
        super().__init__(maxlen=capacity)
    
    def add(self, state, actions, rewards, next_state, done):
        self.append((state, actions, rewards, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.FloatTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones, dtype=np.float32)).to(device)
        )

class MADDPG:
    def __init__(self, env, gamma=0.99, noise=1, noise_decay=0.9995, epsilon=1, epsilon_decay=0.9995, actor_lr=1e-4, critic_lr=1e-3):
        self.env = env
        self.num_agents = env.total_agents
        self.state_dim = len(env.reset())
        self.action_dim = ACTION_DIMENSION
        
        # Initialize networks
        self.actors = [Actor(self.state_dim, self.action_dim).to(device) for _ in range(self.num_agents)]
        self.critics = [Critic(self.state_dim, self.action_dim, self.num_agents).to(device) for _ in range(self.num_agents)]
        self.target_actors = [Actor(self.state_dim, self.action_dim).to(device) for _ in range(self.num_agents)]
        self.target_critics = [Critic(self.state_dim, self.action_dim, self.num_agents).to(device) for _ in range(self.num_agents)]
        
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

    # def update_tau(self, new_tau="Not Change"):
    #     if new_tau=="Not Change":
    #         self.tau *= self.tau_decay
    #     else:
    #         self.tau = new_tau

    def update_noise(self):
        self.noise_scale *= self.noise_decay

    def update_epsilon(self):
        if self.epsilon==0:
            return
        self.epsilon *= self.epsilon_decay
    
    def act(self, state):
        # state = noise_mask(state)
        actions = []
        for i in range(self.num_agents):
            # if i >= self.num_agents//2:
            #     action = np.array([0, 0, -1])
            #     actions.append(action)
            if np.random.random() < self.epsilon:
                action = np.array(self.env.induce_step(i))
                actions.append(action)
            else:
                obs = self.env._get_obs(i)
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = self.actors[i](obs_tensor, noise_scale=self.noise_scale)[0].squeeze().detach().numpy()
                actions.append(action)
        # print(actions)
        return actions
        # return actions.detach().numpy()
    
    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return 0, 0

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Ensure that all tensors are on the correct device
        states = states.to(device)
        next_states = next_states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)

        actor_losses = []
        critic_losses = []

        for i in range(self.num_agents):
            
            # Update Critic
            with torch.no_grad():
                target_acts = torch.cat([
                    self.target_actors[j](next_states).detach().to(device) for j in range(self.num_agents)
                ], dim=-1)
                target_q = self.target_critics[i](next_states, target_acts)
                target_q = rewards[:, i].unsqueeze(1) + (1 - dones) * self.gamma * target_q

            current_acts = []
            for j in range(self.num_agents):
                current_acts.append(self.actors[j](states).detach() if j != i else self.actors[j](states))
            current_acts = torch.cat(current_acts, dim=1).to(device)
            current_q = self.critics[i](states, current_acts)
            critic_loss = nn.MSELoss()(current_q, target_q)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), 0.5)
            self.critic_optimizers[i].step()

            # Update Actor
            actor_acts = []
            for j in range(self.num_agents):
                actor_acts.append(self.actors[j](states).detach() if j != i else self.actors[j](states))
            actor_acts = torch.cat(actor_acts, dim=1).to(device)
            actor_loss = -self.critics[i](states, actor_acts).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optimizers[i].step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

            # Soft update target networks
            for target, param in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                target.data.copy_(self.noise_scale * param + (1 - self.noise_scale) * target)
            for target, param in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                target.data.copy_(self.noise_scale * param + (1 - self.noise_scale) * target)


        if len(actor_losses)*len(critic_losses)==0:
            return 0,0
        
        return np.mean(actor_losses), np.mean(critic_losses)

def train(env, episodes=3000, max_steps=200, batch_size=256, is_render=False, task_code="test"):

    print("Using device:", device)

    agent = MADDPG(env,
                   gamma=0.99,
                   noise=0.4,
                   noise_decay=0.999,
                   epsilon=0.6,
                   epsilon_decay=0.999,
                   actor_lr=2.5e-4,
                   critic_lr=1e-3
                   )
    
    rewards_log = []
    episode_rewards = []
    periodical_rewards = []

    fixed_team = 1  # 假设蓝队（teamcode=1）为固定方
    num_agents = env.total_agents

    uniform_path = Path("uniform") / task_code
    if not os.path.exists(uniform_path):
        os.mkdir(uniform_path)
    
    for ep in range(episodes):

        ep_start_time = time.time()

        state = env.reset()
        total_rewards = np.zeros(agent.num_agents)

        a_loss_episode = []
        c_loss_episode = []

        model_save_path = uniform_path / f"model_ep{ep}.pth"
        log_save_path = uniform_path / "log.txt"
        record_save_path = uniform_path / f"record_part_{ep//100}.jsonl"
        
        for _ in range(max_steps):
            actions = []
            for i in range(num_agents):
                if env.drones[i].teamcode == fixed_team:
                    action = np.random.uniform(-1, 1, 3)  # 随机策略
                else:
                    obs = env._get_obs(i)
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action = agent.actors[i](obs_tensor, training=True)[0].detach().cpu().numpy()
                actions.append(action)

            next_state, rewards, done, _ = env.step(actions)
            agent.memory.add(state, actions, rewards, next_state, done)
            # agent.update(batch_size)
            state = next_state

            if is_render:
                env.render()
            
            # agent.memory.add(state, actions, rewards, next_state, done)
            al, cl = agent.update(batch_size)
            a_loss_episode.append(al)
            c_loss_episode.append(cl)
            
            total_rewards += rewards
            state = next_state
            
            if done:
                break
        
        ep_end_time = time.time()
        time_cosumed = ep_end_time - ep_start_time

        avg_reward = total_rewards.mean()
        rewards_log.append(avg_reward)
        log_text = f"Episode {ep+1}, Reward:{avg_reward:.2f}, Noise:{agent.noise_scale:.3f}, epsilon:{agent.epsilon: .3f}, aloss:{np.mean(a_loss_episode): .3f}, closs:{np.mean(c_loss_episode): .3f}, time:{time_cosumed: .2f}"
        print(log_text)
        with open(log_save_path, "a") as logfile:
            logfile.write(log_text+"\n")
            logfile.close()

        if ep < 1000:
            agent.update_noise()
            agent.update_epsilon()

        episode_rewards.append(total_rewards[:].mean())
        periodical_rewards.append(total_rewards[:].mean())

        env.save_and_clear(ep, record_save_path)
        
        if ep % 100 == 0:
            torch.save(agent.actors[0].state_dict(), model_save_path)

    
    return rewards_log

if __name__ == "__main__":
    env = BattleEnv(red_agents=5, blue_agents=5)
    rewards = train(env)