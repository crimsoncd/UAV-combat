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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

ACTION_DIMENSION = 3


def gumbel_softmax(logits, tau=1.0, hard=False):
    gumbels = -torch.empty_like(logits).exponential_().log()
    y = logits + gumbels
    y = F.softmax(y / tau, dim=-1)
    if hard:
        y_hard = torch.zeros_like(y).scatter_(-1, torch.argmax(y, dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
    return y




# Actor网络：输出连续动作（移动方向+射击概率）
class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(state_dim, state_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_DIMENSION),
            nn.Tanh() 
        )
        self.to(device)

    def forward(self, state, tau=0.08, hard=True):
        # r = np.random.rand()
        # state = state.to(device)
        # out, _ = self.lstm(state)
        # logits = self.net(out)
        # logits = self.net(state)
        # action = gumbel_softmax(logits, tau=tau, hard=hard)
        # out = torch.argmax(action, dim=-1, keepdim=True)  # 训练时返回soft采样，测试时返回hard
        # if r<tau:
        #     return torch.randint(0, 12, out.shape).cpu()
        # return out.cpu()
        return self.net(state).cpu()
    

# Critic网络：输入全局状态+所有智能体动作
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents, hidden_dim=256):
        super(Critic, self).__init__()
        input_dim = state_dim + action_dim * num_agents
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
    
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
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
    def __init__(self, env, gamma=0.99, tau=1, tau_decay=0.995, epsilon=1, epsilon_decay=0.9995, actor_lr=1e-4, critic_lr=1e-3):
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
        self.tau = tau
        self.tau_decay = tau_decay
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.noise_scale = 5
        self.noise_decay = 0.9995
        self.noise_decrease = 0.01/200
        self.memory = ReplayBuffer(100000)

    def update_tau(self, new_tau = 0):
        if new_tau>0:
            self.tau = new_tau
        else:
            self.tau *= self.tau_decay

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
    
    def act(self, state):
        # state = noise_mask(state)
        actions = []
        explore = np.random.rand()
        # state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # for i in range(self.num_agents):
        #     if explore <= self.epsilon:
        #         action = self.env.induce_step(i)
        #     else:
        #         obs = self.env._get_obs(i)
        #         obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        #         action = self.actors[i](obs_tensor, tau=self.tau)[0].squeeze()
        #     actions.append(action)
        # print(actions)
        for i in range(self.num_agents):
            obs = self.env._get_obs(i)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action = self.actors[i](obs_tensor, tau=self.tau)[0].squeeze().detach().numpy()
            actions.append(action)
        # print(actions)
        return actions
        # return actions.detach().numpy()
    
    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return 0,0
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = states.view(batch_size, -1).to(device)
        next_states = next_states.view(batch_size, -1).to(device)
        actions = actions.view(batch_size, -1).to(device)

        actor_losses = []
        critic_losses = []
        
        for i in range(self.num_agents):
            # Update Critic
            with torch.no_grad():
                target_acts = torch.cat([
                    self.target_actors[j](next_states).to(device) if j == i 
                    else self.target_actors[j](next_states).detach().to(device)
                    for j in range(self.num_agents)
                ], dim=-1)
                # print(next_states.shape, target_acts.shape)
                target_q = self.target_critics[i](next_states, target_acts)
                target_q = rewards[:, i].unsqueeze(1) + (1 - dones) * self.gamma * target_q
            
            current_acts = []
            for j in range(self.num_agents):
                if j == i:
                    current_act = self.actors[j](states)
                else:
                    current_act = self.actors[j](states).detach()
                current_acts.append(current_act.to(device))
            current_acts = torch.cat(current_acts, dim=1)
            current_q = self.critics[i](states, current_acts)
            critic_loss = nn.MSELoss()(current_q, target_q)
            
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), 0.5)
            self.critic_optimizers[i].step()
            
            # Update Actor
            actor_acts = []
            for j in range(self.num_agents):
                if j == i:
                    actor_act = self.actors[j](states)
                else:
                    actor_act = self.actors[j](states).detach()
                actor_acts.append(actor_act.to(device))
            actor_acts = torch.cat(actor_acts, dim=1)
            actor_loss = -self.critics[i](states, actor_acts).mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optimizers[i].step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            
            # Soft update target networks
            for target, param in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                target.data.copy_(self.tau * param + (1 - self.tau) * target)
            for target, param in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                target.data.copy_(self.tau * param + (1 - self.tau) * target)
        
        # self.noise_scale *= self.noise_decay
        # self.noise_scale -= self.noise_decrease
        # self.noise_scale = max(self.noise_scale, 0)
        # print(np.mean(actor_losses), np.mean(critic_losses))
        if len(actor_losses)*len(critic_losses)==0:
            return 0,0
        return np.mean(actor_losses), np.mean(critic_losses)

def train(env, episodes=3000, max_steps=200, batch_size=256, is_render=False, task_code="test"):

    print("Using device:", device)

    agent = MADDPG(env)
    rewards_log = []
    episode_rewards = []
    periodical_rewards = []

    model_save_path = "models/actor_" + task_code
    log_save_path = "pics/totallog_" + task_code + ".txt"
    record_save_path = "record/record_" + task_code + ".jsonl"
    
    
    for ep in range(episodes):

        ep_start_time = time.time()

        state = env.reset()
        total_rewards = np.zeros(agent.num_agents)

        a_loss_episode = []
        c_loss_episode = []
        film_record = []
        
        for _ in range(max_steps):
            actions = agent.act(state)
            # print(actions)
            next_state, rewards, done, _ = env.step(actions)
            # next_state, rewards, done, _ = env.step([a.cpu() if type(a)!=int else a for a in actions])
            film_record.append(next_state.tolist())

            if is_render:
                env.render()
            
            agent.memory.add(state, actions, rewards, next_state, done)
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
        log_text = f"Episode {ep+1}, Reward:{avg_reward:.2f}, Noise:{agent.tau:.3f}, epsilon:{agent.epsilon: .3f}, aloss:{np.mean(a_loss_episode): .3f}, closs:{np.mean(c_loss_episode): .3f}, time:{time_cosumed: .2f}"
        print(log_text)
        with open(log_save_path, "a") as logfile:
            logfile.write(log_text+"\n")
            logfile.close()

        change_tau = 0.08 if ep>500 else 0
        agent.update_tau(new_tau=change_tau)
        agent.update_epsilon()

        episode_rewards.append(total_rewards[:].mean())
        periodical_rewards.append(total_rewards[:].mean())

        env.save_and_clear(ep, record_save_path)
        
        if ep % 100 == 0:
            torch.save(agent.actors[0].state_dict(), model_save_path + f"_ep{ep}.pth")

    
    return rewards_log

if __name__ == "__main__":
    env = BattleEnv(red_agents=5, blue_agents=5)
    rewards = train(env)