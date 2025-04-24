# import matplotlib.pyplot as plt
# import numpy as np


# def dense_action_reward(epoch=1000, step=200):

#     r = np.zeros(epoch)
#     reward_scale = 1

#     for i in range(epoch):
#         epoch_rewards = []
#         for j in range(step):
#             epoch_reward = 0
#             action = np.random.random(3) * 2 - 1
#             epoch_reward += reward_scale * (1 - abs(action[0]))
#             epoch_reward += reward_scale * (1 - abs(action[1]))
#             epoch_reward += reward_scale * (1 - abs(action[2]))
#             epoch_reward -= reward_scale * 1.5
#             epoch_rewards.append(epoch_reward)
#         r[i] += np.mean(epoch_rewards)

#     return r


# if __name__=="__main__":

#     test_r = dense_action_reward()
#     range_r = [i for i in range(len(test_r))]
#     plt.plot(range_r, test_r)
#     plt.show()


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

NUM_AGENTS = 2
OBS_DIM = 3
ACT_DIM = 3


# ==== 环境定义 ====
class SimpleEnv:
    def reset(self):
        return [np.zeros(OBS_DIM) for _ in range(NUM_AGENTS)]

    def step(self, actions):
        # rewards = [3 - np.sum(np.abs(a)) for a in actions]  # reward 越接近0越高
        rewards = [3 - np.sum(np.sqrt(np.abs(a))) for a in actions]
        next_obs = [np.zeros(OBS_DIM) for _ in range(NUM_AGENTS)]
        done = False
        return next_obs, rewards, done, {}


# ==== 网络结构 ====
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(OBS_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, ACT_DIM),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(NUM_AGENTS * (OBS_DIM + ACT_DIM), 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs, acts):
        x = torch.cat([obs, acts], dim=-1)
        return self.fc(x)


# ==== 经验回放 ====
class ReplayBuffer:
    def __init__(self, maxlen=10000):
        self.buffer = []

    def add(self, obs, act, rew, next_obs):
        self.buffer.append((obs, act, rew, next_obs))
        if len(self.buffer) > 10000:
            self.buffer.pop(0)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs = zip(*batch)
        return (
            torch.FloatTensor(obs).to(device),      # [B, N, obs_dim]
            torch.FloatTensor(act).to(device),
            torch.FloatTensor(rew).to(device),
            torch.FloatTensor(next_obs).to(device)
        )


# ==== MADDPG ====
class MADDPG:
    def __init__(self, gamma=0.95):
        self.actors = [Actor().to(device) for _ in range(NUM_AGENTS)]
        self.critics = [Critic().to(device) for _ in range(NUM_AGENTS)]
        self.target_actors = [Actor().to(device) for _ in range(NUM_AGENTS)]
        self.target_critics = [Critic().to(device) for _ in range(NUM_AGENTS)]
        self.actor_opt = [optim.Adam(a.parameters(), lr=1e-3) for a in self.actors]
        self.critic_opt = [optim.Adam(c.parameters(), lr=1e-3) for c in self.critics]
        self.gamma = gamma
        self.buffer = ReplayBuffer()

        for i in range(NUM_AGENTS):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

    def select_action(self, obs, noise_scale = 1):
        obs_tensor = [torch.FloatTensor(o).unsqueeze(0).to(device) for o in obs]
        acts = [self.actors[i](obs_tensor[i]).detach().cpu().numpy()[0] for i in range(NUM_AGENTS)]
        noise = np.random.random(3) * 2 - 1
        noise *= noise_scale
        acts += noise
        acts = np.clip(acts, -1, 1)
        return acts

    def update(self, batch_size=64):
        if len(self.buffer.buffer) < batch_size:
            return

        obs, act, rew, next_obs = self.buffer.sample(batch_size)
        obs = obs.view(batch_size, -1)
        act = act.view(batch_size, -1)
        next_obs = next_obs.view(batch_size, -1)

        for i in range(NUM_AGENTS):
            with torch.no_grad():
                next_acts = [self.target_actors[j](next_obs[:, j * OBS_DIM:(j + 1) * OBS_DIM]) for j in range(NUM_AGENTS)]
                next_acts_cat = torch.cat(next_acts, dim=1)
                target_q = self.target_critics[i](next_obs, next_acts_cat)
                y = rew[:, i].unsqueeze(1) + self.gamma * target_q

            q = self.critics[i](obs, act)
            critic_loss = nn.MSELoss()(q, y)

            self.critic_opt[i].zero_grad()
            critic_loss.backward()
            self.critic_opt[i].step()

            curr_acts = [self.actors[j](obs[:, j * OBS_DIM:(j + 1) * OBS_DIM]) for j in range(NUM_AGENTS)]
            curr_acts_cat = torch.cat(curr_acts, dim=1)
            actor_loss = -self.critics[i](obs, curr_acts_cat).mean()

            self.actor_opt[i].zero_grad()
            actor_loss.backward()
            self.actor_opt[i].step()

            # Soft update
            for param, target in zip(self.actors[i].parameters(), self.target_actors[i].parameters()):
                target.data.copy_(0.01 * param + 0.99 * target)
            for param, target in zip(self.critics[i].parameters(), self.target_critics[i].parameters()):
                target.data.copy_(0.01 * param + 0.99 * target)


# ==== 训练 ====
env = SimpleEnv()
agent = MADDPG()
episodes = 500
print_every = 10

for ep in range(episodes):
    obs = env.reset()
    total_reward = 0
    for _ in range(50):
        n_scale = max(0, 0.4-0.002*ep)
        act = agent.select_action(obs, n_scale)
        next_obs, rew, done, _ = env.step(act)
        agent.buffer.add(obs, act, rew, next_obs)
        obs = next_obs
        total_reward += np.mean(rew)
        agent.update()

    if ep % print_every == 0:
        print(f"[Ep {ep}] Avg Reward: {total_reward / 50:.3f}  | Actions: {[round(a[0], 3) for a in act]}")



"""
[Ep 10] Avg Reward: 0.860  | Actions: [np.float32(-0.51), np.float32(0.984)]
[Ep 20] Avg Reward: 2.488  | Actions: [np.float32(-0.253), np.float32(0.115)]
[Ep 30] Avg Reward: 2.522  | Actions: [np.float32(-0.271), np.float32(0.072)]
[Ep 40] Avg Reward: 2.534  | Actions: [np.float32(-0.255), np.float32(0.05)]
[Ep 50] Avg Reward: 2.548  | Actions: [np.float32(-0.281), np.float32(0.129)]
[Ep 60] Avg Reward: 2.592  | Actions: [np.float32(-0.267), np.float32(0.138)]
[Ep 70] Avg Reward: 2.602  | Actions: [np.float32(-0.243), np.float32(0.185)]
[Ep 80] Avg Reward: 2.613  | Actions: [np.float32(-0.226), np.float32(0.179)]
[Ep 90] Avg Reward: 2.622  | Actions: [np.float32(-0.206), np.float32(0.193)]
[Ep 100] Avg Reward: 2.622  | Actions: [np.float32(-0.231), np.float32(0.164)]
[Ep 110] Avg Reward: 2.627  | Actions: [np.float32(-0.194), np.float32(0.193)]
[Ep 120] Avg Reward: 2.626  | Actions: [np.float32(-0.213), np.float32(0.193)]
[Ep 130] Avg Reward: 2.625  | Actions: [np.float32(-0.215), np.float32(0.192)]
[Ep 140] Avg Reward: 2.626  | Actions: [np.float32(-0.208), np.float32(0.19)]
[Ep 150] Avg Reward: 2.627  | Actions: [np.float32(-0.216), np.float32(0.174)]
[Ep 160] Avg Reward: 2.628  | Actions: [np.float32(-0.2), np.float32(0.226)]
[Ep 170] Avg Reward: 2.629  | Actions: [np.float32(-0.197), np.float32(0.216)]
[Ep 180] Avg Reward: 2.630  | Actions: [np.float32(-0.209), np.float32(0.197)]
[Ep 190] Avg Reward: 2.632  | Actions: [np.float32(-0.212), np.float32(0.183)]
[Ep 200] Avg Reward: 2.630  | Actions: [np.float32(-0.208), np.float32(0.198)]
[Ep 210] Avg Reward: 2.632  | Actions: [np.float32(-0.212), np.float32(0.197)]
[Ep 220] Avg Reward: 2.635  | Actions: [np.float32(-0.217), np.float32(0.188)]
[Ep 230] Avg Reward: 2.636  | Actions: [np.float32(-0.216), np.float32(0.195)]
[Ep 240] Avg Reward: 2.639  | Actions: [np.float32(-0.22), np.float32(0.179)]
[Ep 250] Avg Reward: 2.641  | Actions: [np.float32(-0.214), np.float32(0.2)]
[Ep 260] Avg Reward: 2.642  | Actions: [np.float32(-0.214), np.float32(0.209)]
[Ep 270] Avg Reward: 2.644  | Actions: [np.float32(-0.219), np.float32(0.205)]
[Ep 280] Avg Reward: 2.644  | Actions: [np.float32(-0.213), np.float32(0.215)]
[Ep 290] Avg Reward: 2.645  | Actions: [np.float32(-0.215), np.float32(0.216)]
[Ep 300] Avg Reward: 2.646  | Actions: [np.float32(-0.223), np.float32(0.193)]
[Ep 310] Avg Reward: 2.648  | Actions: [np.float32(-0.226), np.float32(0.194)]
[Ep 320] Avg Reward: 2.647  | Actions: [np.float32(-0.221), np.float32(0.206)]
[Ep 330] Avg Reward: 2.649  | Actions: [np.float32(-0.213), np.float32(0.215)]
[Ep 340] Avg Reward: 2.651  | Actions: [np.float32(-0.214), np.float32(0.214)]
[Ep 350] Avg Reward: 2.646  | Actions: [np.float32(-0.233), np.float32(0.174)]
[Ep 370] Avg Reward: 2.652  | Actions: [np.float32(-0.222), np.float32(0.209)]
[Ep 380] Avg Reward: 2.651  | Actions: [np.float32(-0.229), np.float32(0.184)]
[Ep 390] Avg Reward: 2.654  | Actions: [np.float32(-0.224), np.float32(0.198)]
[Ep 400] Avg Reward: 2.653  | Actions: [np.float32(-0.22), np.float32(0.198)]
[Ep 410] Avg Reward: 2.652  | Actions: [np.float32(-0.219), np.float32(0.211)]
[Ep 420] Avg Reward: 2.651  | Actions: [np.float32(-0.226), np.float32(0.191)]
[Ep 430] Avg Reward: 2.655  | Actions: [np.float32(-0.228), np.float32(0.201)]
[Ep 440] Avg Reward: 2.658  | Actions: [np.float32(-0.218), np.float32(0.214)]
[Ep 450] Avg Reward: 2.657  | Actions: [np.float32(-0.233), np.float32(0.192)]
[Ep 460] Avg Reward: 2.661  | Actions: [np.float32(-0.23), np.float32(0.206)]
[Ep 470] Avg Reward: 2.660  | Actions: [np.float32(-0.215), np.float32(0.216)]
[Ep 480] Avg Reward: 2.660  | Actions: [np.float32(-0.228), np.float32(0.216)]
[Ep 490] Avg Reward: 2.659  | Actions: [np.float32(-0.22), np.float32(0.211)]
PS D:\Develop\Codes\UAV\self-ju\train\I_GitRemote\UAV-combat> & D:/Develop/anaconda3/envs/SE/python.exe d:/Develop/Codes/UAV/self-ju/train/I_GitRemote/UAV-combat/dense_test.py
[Ep 0] Avg Reward: 2.631  | Actions: [np.float32(-0.175), np.float32(0.191)]
d:\Develop\Codes\UAV\self-ju\train\I_GitRemote\UAV-combat\dense_test.py:103: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\utils\tensor_new.cpp:257.)
  torch.FloatTensor(obs).to(device),      # [B, N, obs_dim]
[Ep 10] Avg Reward: 1.193  | Actions: [np.float32(-0.296), np.float32(0.984)]
[Ep 20] Avg Reward: 2.420  | Actions: [np.float32(-0.223), np.float32(0.236)]
[Ep 30] Avg Reward: 2.544  | Actions: [np.float32(-0.218), np.float32(0.219)]
[Ep 40] Avg Reward: 2.584  | Actions: [np.float32(-0.215), np.float32(0.219)]
[Ep 50] Avg Reward: 2.600  | Actions: [np.float32(-0.217), np.float32(0.234)]
[Ep 60] Avg Reward: 2.594  | Actions: [np.float32(-0.239), np.float32(0.236)]
[Ep 70] Avg Reward: 2.590  | Actions: [np.float32(-0.259), np.float32(0.23)]
[Ep 80] Avg Reward: 2.595  | Actions: [np.float32(-0.263), np.float32(0.225)]
[Ep 90] Avg Reward: 2.602  | Actions: [np.float32(-0.242), np.float32(0.213)]
[Ep 100] Avg Reward: 2.601  | Actions: [np.float32(-0.212), np.float32(0.276)]
[Ep 110] Avg Reward: 2.614  | Actions: [np.float32(-0.21), np.float32(0.226)]
[Ep 120] Avg Reward: 2.614  | Actions: [np.float32(-0.235), np.float32(0.209)]
[Ep 130] Avg Reward: 2.617  | Actions: [np.float32(-0.2), np.float32(0.25)]
[Ep 140] Avg Reward: 2.615  | Actions: [np.float32(-0.209), np.float32(0.236)]
[Ep 150] Avg Reward: 2.622  | Actions: [np.float32(-0.204), np.float32(0.223)]
[Ep 160] Avg Reward: 2.625  | Actions: [np.float32(-0.183), np.float32(0.23)]
[Ep 170] Avg Reward: 2.625  | Actions: [np.float32(-0.179), np.float32(0.234)]
[Ep 180] Avg Reward: 2.625  | Actions: [np.float32(-0.179), np.float32(0.233)]
[Ep 190] Avg Reward: 2.628  | Actions: [np.float32(-0.184), np.float32(0.245)]
[Ep 200] Avg Reward: 2.629  | Actions: [np.float32(-0.184), np.float32(0.232)]
[Ep 210] Avg Reward: 2.633  | Actions: [np.float32(-0.189), np.float32(0.231)]
[Ep 220] Avg Reward: 2.634  | Actions: [np.float32(-0.183), np.float32(0.23)]
[Ep 230] Avg Reward: 2.629  | Actions: [np.float32(-0.22), np.float32(0.195)]
[Ep 240] Avg Reward: 2.638  | Actions: [np.float32(-0.178), np.float32(0.218)]
[Ep 250] Avg Reward: 2.637  | Actions: [np.float32(-0.177), np.float32(0.236)]
[Ep 260] Avg Reward: 2.642  | Actions: [np.float32(-0.184), np.float32(0.23)]
[Ep 270] Avg Reward: 2.645  | Actions: [np.float32(-0.18), np.float32(0.24)]
[Ep 280] Avg Reward: 2.642  | Actions: [np.float32(-0.181), np.float32(0.247)]
[Ep 290] Avg Reward: 2.642  | Actions: [np.float32(-0.191), np.float32(0.233)]
[Ep 300] Avg Reward: 2.648  | Actions: [np.float32(-0.18), np.float32(0.256)]
[Ep 310] Avg Reward: 2.645  | Actions: [np.float32(-0.225), np.float32(0.185)]
[Ep 320] Avg Reward: 2.648  | Actions: [np.float32(-0.215), np.float32(0.202)]
[Ep 330] Avg Reward: 2.651  | Actions: [np.float32(-0.212), np.float32(0.209)]
[Ep 340] Avg Reward: 2.653  | Actions: [np.float32(-0.189), np.float32(0.241)]
[Ep 350] Avg Reward: 2.651  | Actions: [np.float32(-0.205), np.float32(0.222)]
[Ep 370] Avg Reward: 2.660  | Actions: [np.float32(-0.194), np.float32(0.229)]
[Ep 380] Avg Reward: 2.659  | Actions: [np.float32(-0.229), np.float32(0.175)]
[Ep 390] Avg Reward: 2.662  | Actions: [np.float32(-0.199), np.float32(0.239)]
[Ep 400] Avg Reward: 2.664  | Actions: [np.float32(-0.203), np.float32(0.226)]
[Ep 410] Avg Reward: 2.660  | Actions: [np.float32(-0.229), np.float32(0.201)]
[Ep 420] Avg Reward: 2.666  | Actions: [np.float32(-0.207), np.float32(0.221)]
[Ep 430] Avg Reward: 2.667  | Actions: [np.float32(-0.208), np.float32(0.231)]
[Ep 440] Avg Reward: 2.669  | Actions: [np.float32(-0.213), np.float32(0.217)]
[Ep 450] Avg Reward: 2.667  | Actions: [np.float32(-0.216), np.float32(0.224)]
[Ep 460] Avg Reward: 2.668  | Actions: [np.float32(-0.217), np.float32(0.227)]
[Ep 470] Avg Reward: 2.672  | Actions: [np.float32(-0.231), np.float32(0.19)]
[Ep 480] Avg Reward: 2.673  | Actions: [np.float32(-0.22), np.float32(0.217)]
[Ep 490] Avg Reward: 2.674  | Actions: [np.float32(-0.22), np.float32(0.213)]
PS D:\Develop\Codes\UAV\self-ju\train\I_GitRemote\UAV-combat> & D:/Develop/anaconda3/envs/SE/python.exe d:/Develop/Codes/UAV/self-ju/train/I_GitRemote/UAV-combat/dense_test.py
[Ep 0] Avg Reward: 2.324  | Actions: [np.float64(0.16), np.float64(0.527)]
d:\Develop\Codes\UAV\self-ju\train\I_GitRemote\UAV-combat\dense_test.py:103: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\utils\tensor_new.cpp:257.)
  torch.FloatTensor(obs).to(device),      # [B, N, obs_dim]
[Ep 10] Avg Reward: 1.267  | Actions: [np.float64(-1.0), np.float64(-0.677)]
[Ep 20] Avg Reward: 2.291  | Actions: [np.float64(-0.242), np.float64(0.351)]
[Ep 30] Avg Reward: 2.350  | Actions: [np.float64(-0.58), np.float64(0.038)]
[Ep 40] Avg Reward: 2.346  | Actions: [np.float64(-0.346), np.float64(0.262)]
[Ep 50] Avg Reward: 2.338  | Actions: [np.float64(-0.257), np.float64(0.352)]
[Ep 60] Avg Reward: 2.380  | Actions: [np.float64(-0.517), np.float64(0.082)]
[Ep 70] Avg Reward: 2.433  | Actions: [np.float64(-0.597), np.float64(-0.013)]
[Ep 80] Avg Reward: 2.450  | Actions: [np.float64(-0.542), np.float64(0.038)]
[Ep 90] Avg Reward: 2.528  | Actions: [np.float64(-0.381), np.float64(0.173)]
[Ep 100] Avg Reward: 2.511  | Actions: [np.float64(-0.367), np.float64(0.18)]
[Ep 110] Avg Reward: 2.543  | Actions: [np.float64(0.133), np.float64(0.588)]
[Ep 120] Avg Reward: 2.577  | Actions: [np.float64(-0.025), np.float64(0.412)]
[Ep 130] Avg Reward: 2.628  | Actions: [np.float64(-0.087), np.float64(0.344)]
[Ep 140] Avg Reward: 2.638  | Actions: [np.float64(0.059), np.float64(0.505)]
[Ep 150] Avg Reward: 2.683  | Actions: [np.float64(-0.057), np.float64(0.376)]
[Ep 160] Avg Reward: 2.689  | Actions: [np.float64(0.029), np.float64(0.465)]
[Ep 170] Avg Reward: 2.721  | Actions: [np.float64(-0.068), np.float64(0.362)]
[Ep 180] Avg Reward: 2.741  | Actions: [np.float64(0.026), np.float64(0.45)]
[Ep 190] Avg Reward: 2.766  | Actions: [np.float64(-0.002), np.float64(0.43)]
[Ep 200] Avg Reward: 2.780  | Actions: [np.float64(-0.011), np.float64(0.414)]
[Ep 210] Avg Reward: 2.783  | Actions: [np.float64(-0.075), np.float64(0.345)]
[Ep 220] Avg Reward: 2.784  | Actions: [np.float64(-0.049), np.float64(0.37)]
[Ep 230] Avg Reward: 2.785  | Actions: [np.float64(-0.036), np.float64(0.381)]
[Ep 240] Avg Reward: 2.787  | Actions: [np.float64(-0.079), np.float64(0.332)]
[Ep 250] Avg Reward: 2.789  | Actions: [np.float64(-0.093), np.float64(0.318)]
[Ep 260] Avg Reward: 2.791  | Actions: [np.float64(-0.04), np.float64(0.368)]
[Ep 270] Avg Reward: 2.791  | Actions: [np.float64(-0.124), np.float64(0.282)]
[Ep 280] Avg Reward: 2.795  | Actions: [np.float64(-0.037), np.float64(0.365)]
[Ep 290] Avg Reward: 2.795  | Actions: [np.float64(-0.078), np.float64(0.321)]
[Ep 300] Avg Reward: 2.797  | Actions: [np.float64(-0.128), np.float64(0.265)]
[Ep 310] Avg Reward: 2.798  | Actions: [np.float64(-0.104), np.float64(0.288)]
[Ep 320] Avg Reward: 2.801  | Actions: [np.float64(-0.081), np.float64(0.308)]
[Ep 330] Avg Reward: 2.802  | Actions: [np.float64(-0.096), np.float64(0.289)]
[Ep 340] Avg Reward: 2.804  | Actions: [np.float64(-0.097), np.float64(0.286)]
[Ep 350] Avg Reward: 2.806  | Actions: [np.float64(-0.135), np.float64(0.248)]
[Ep 370] Avg Reward: 2.809  | Actions: [np.float64(-0.125), np.float64(0.254)]
[Ep 380] Avg Reward: 2.810  | Actions: [np.float64(-0.139), np.float64(0.237)]
[Ep 390] Avg Reward: 2.811  | Actions: [np.float64(-0.122), np.float64(0.253)]
[Ep 400] Avg Reward: 2.812  | Actions: [np.float64(-0.112), np.float64(0.261)]
[Ep 410] Avg Reward: 2.812  | Actions: [np.float64(-0.133), np.float64(0.239)]
[Ep 420] Avg Reward: 2.813  | Actions: [np.float64(-0.113), np.float64(0.256)]
[Ep 430] Avg Reward: 2.815  | Actions: [np.float64(-0.108), np.float64(0.26)]
[Ep 440] Avg Reward: 2.815  | Actions: [np.float64(-0.104), np.float64(0.263)]
[Ep 450] Avg Reward: 2.816  | Actions: [np.float64(-0.086), np.float64(0.276)]
[Ep 460] Avg Reward: 2.817  | Actions: [np.float64(-0.131), np.float64(0.231)]
[Ep 470] Avg Reward: 2.818  | Actions: [np.float64(-0.117), np.float64(0.241)]
[Ep 480] Avg Reward: 2.819  | Actions: [np.float64(-0.119), np.float64(0.24)]
[Ep 490] Avg Reward: 2.820  | Actions: [np.float64(-0.122), np.float64(0.236)]
PS D:\Develop\Codes\UAV\self-ju\train\I_GitRemote\UAV-combat> & D:/Develop/anaconda3/envs/SE/python.exe d:/Develop/Codes/UAV/self-ju/train/I_GitRemote/UAV-combat/dense_test.py
[Ep 0] Avg Reward: 2.778  | Actions: [np.float64(0.16), np.float64(0.527)]
d:\Develop\Codes\UAV\self-ju\train\I_GitRemote\UAV-combat\dense_test.py:104: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\utils\tensor_new.cpp:257.)
  torch.FloatTensor(obs).to(device),      # [B, N, obs_dim]
[Ep 10] Avg Reward: 1.490  | Actions: [np.float64(-1.0), np.float64(-0.807)]
[Ep 20] Avg Reward: 2.726  | Actions: [np.float64(0.016), np.float64(0.503)]
[Ep 30] Avg Reward: 2.768  | Actions: [np.float64(-0.435), np.float64(0.085)]
[Ep 40] Avg Reward: 2.768  | Actions: [np.float64(-0.181), np.float64(0.402)]
[Ep 50] Avg Reward: 2.788  | Actions: [np.float64(-0.066), np.float64(0.517)]
[Ep 60] Avg Reward: 2.822  | Actions: [np.float64(-0.304), np.float64(0.248)]
[Ep 70] Avg Reward: 2.836  | Actions: [np.float64(-0.41), np.float64(0.146)]
[Ep 80] Avg Reward: 2.850  | Actions: [np.float64(-0.358), np.float64(0.193)]
[Ep 90] Avg Reward: 2.867  | Actions: [np.float64(-0.193), np.float64(0.351)]
[Ep 100] Avg Reward: 2.863  | Actions: [np.float64(-0.176), np.float64(0.353)]
[Ep 110] Avg Reward: 2.871  | Actions: [np.float64(-0.031), np.float64(0.481)]
[Ep 120] Avg Reward: 2.888  | Actions: [np.float64(-0.181), np.float64(0.334)]
[Ep 130] Avg Reward: 2.895  | Actions: [np.float64(-0.239), np.float64(0.253)]
[Ep 140] Avg Reward: 2.903  | Actions: [np.float64(-0.13), np.float64(0.377)]
[Ep 150] Avg Reward: 2.913  | Actions: [np.float64(-0.275), np.float64(0.231)]
[Ep 160] Avg Reward: 2.919  | Actions: [np.float64(-0.158), np.float64(0.305)]
[Ep 170] Avg Reward: 2.921  | Actions: [np.float64(-0.215), np.float64(0.226)]
[Ep 180] Avg Reward: 2.925  | Actions: [np.float64(-0.155), np.float64(0.294)]
[Ep 190] Avg Reward: 2.926  | Actions: [np.float64(-0.187), np.float64(0.265)]
[Ep 200] Avg Reward: 2.924  | Actions: [np.float64(-0.182), np.float64(0.26)]
[Ep 210] Avg Reward: 2.925  | Actions: [np.float64(-0.2), np.float64(0.253)]
[Ep 220] Avg Reward: 2.928  | Actions: [np.float64(-0.214), np.float64(0.236)]
[Ep 230] Avg Reward: 2.931  | Actions: [np.float64(-0.224), np.float64(0.228)]
[Ep 240] Avg Reward: 2.934  | Actions: [np.float64(-0.179), np.float64(0.237)]
[Ep 250] Avg Reward: 2.936  | Actions: [np.float64(-0.078), np.float64(0.276)]
[Ep 260] Avg Reward: 2.937  | Actions: [np.float64(-0.084), np.float64(0.276)]
[Ep 270] Avg Reward: 2.936  | Actions: [np.float64(-0.084), np.float64(0.27)]
[Ep 280] Avg Reward: 2.937  | Actions: [np.float64(-0.091), np.float64(0.268)]
[Ep 290] Avg Reward: 2.940  | Actions: [np.float64(-0.094), np.float64(0.275)]
[Ep 300] Avg Reward: 2.942  | Actions: [np.float64(-0.095), np.float64(0.272)]
[Ep 310] Avg Reward: 2.944  | Actions: [np.float64(-0.101), np.float64(0.267)]
[Ep 320] Avg Reward: 2.945  | Actions: [np.float64(-0.101), np.float64(0.268)]
[Ep 330] Avg Reward: 2.946  | Actions: [np.float64(-0.1), np.float64(0.268)]
[Ep 340] Avg Reward: 2.946  | Actions: [np.float64(-0.106), np.float64(0.261)]
[Ep 350] Avg Reward: 2.948  | Actions: [np.float64(-0.114), np.float64(0.265)]
[Ep 370] Avg Reward: 2.949  | Actions: [np.float64(-0.108), np.float64(0.26)]
[Ep 380] Avg Reward: 2.949  | Actions: [np.float64(-0.11), np.float64(0.256)]
[Ep 390] Avg Reward: 2.950  | Actions: [np.float64(-0.114), np.float64(0.254)]
[Ep 400] Avg Reward: 2.951  | Actions: [np.float64(-0.109), np.float64(0.256)]
[Ep 410] Avg Reward: 2.951  | Actions: [np.float64(-0.109), np.float64(0.255)]
[Ep 420] Avg Reward: 2.952  | Actions: [np.float64(-0.113), np.float64(0.249)]
[Ep 430] Avg Reward: 2.952  | Actions: [np.float64(-0.113), np.float64(0.249)]
[Ep 440] Avg Reward: 2.953  | Actions: [np.float64(-0.114), np.float64(0.245)]
[Ep 450] Avg Reward: 2.954  | Actions: [np.float64(-0.115), np.float64(0.242)]
[Ep 460] Avg Reward: 2.954  | Actions: [np.float64(-0.113), np.float64(0.244)]
[Ep 470] Avg Reward: 2.955  | Actions: [np.float64(-0.112), np.float64(0.243)]
[Ep 480] Avg Reward: 2.955  | Actions: [np.float64(-0.113), np.float64(0.244)]
[Ep 490] Avg Reward: 2.955  | Actions: [np.float64(-0.113), np.float64(0.241)]
PS D:\Develop\Codes\UAV\self-ju\train\I_GitRemote\UAV-combat> & D:/Develop/anaconda3/envs/SE/python.exe d:/Develop/Codes/UAV/self-ju/train/I_GitRemote/UAV-combat/dense_test.py
[Ep 0] Avg Reward: 1.677  | Actions: [np.float64(0.16), np.float64(0.527)]
d:\Develop\Codes\UAV\self-ju\train\I_GitRemote\UAV-combat\dense_test.py:104: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\utils\tensor_new.cpp:257.)
  torch.FloatTensor(obs).to(device),      # [B, N, obs_dim]
[Ep 10] Avg Reward: 0.885  | Actions: [np.float64(-0.998), np.float64(-0.637)]
[Ep 20] Avg Reward: 1.664  | Actions: [np.float64(-0.101), np.float64(0.464)]
[Ep 30] Avg Reward: 1.755  | Actions: [np.float64(-0.43), np.float64(0.141)]
[Ep 40] Avg Reward: 1.669  | Actions: [np.float64(0.088), np.float64(0.686)]
[Ep 50] Avg Reward: 1.685  | Actions: [np.float64(0.18), np.float64(0.792)]
[Ep 60] Avg Reward: 1.696  | Actions: [np.float64(-0.085), np.float64(0.538)]
[Ep 70] Avg Reward: 1.743  | Actions: [np.float64(-0.179), np.float64(0.442)]
[Ep 80] Avg Reward: 1.768  | Actions: [np.float64(-0.124), np.float64(0.499)]
[Ep 90] Avg Reward: 1.899  | Actions: [np.float64(0.047), np.float64(0.67)]
[Ep 100] Avg Reward: 1.825  | Actions: [np.float64(0.048), np.float64(0.677)]
[Ep 110] Avg Reward: 1.881  | Actions: [np.float64(0.179), np.float64(0.807)]
[Ep 120] Avg Reward: 1.914  | Actions: [np.float64(0.026), np.float64(0.65)]
[Ep 130] Avg Reward: 1.986  | Actions: [np.float64(-0.048), np.float64(0.57)]
[Ep 140] Avg Reward: 2.010  | Actions: [np.float64(0.095), np.float64(0.707)]
[Ep 150] Avg Reward: 2.101  | Actions: [np.float64(-0.031), np.float64(0.589)]
[Ep 160] Avg Reward: 2.119  | Actions: [np.float64(0.044), np.float64(0.665)]
[Ep 170] Avg Reward: 2.197  | Actions: [np.float64(-0.041), np.float64(0.583)]
[Ep 180] Avg Reward: 2.278  | Actions: [np.float64(0.038), np.float64(0.652)]
[Ep 190] Avg Reward: 2.377  | Actions: [np.float64(0.012), np.float64(0.631)]
[Ep 200] Avg Reward: 2.497  | Actions: [np.float64(0.003), np.float64(0.622)]
[Ep 210] Avg Reward: 2.501  | Actions: [np.float64(0.002), np.float64(0.617)]
[Ep 220] Avg Reward: 2.505  | Actions: [np.float64(0.004), np.float64(0.621)]
[Ep 230] Avg Reward: 2.504  | Actions: [np.float64(0.003), np.float64(0.615)]
[Ep 240] Avg Reward: 2.502  | Actions: [np.float64(0.002), np.float64(0.612)]
[Ep 250] Avg Reward: 2.507  | Actions: [np.float64(0.001), np.float64(0.614)]
[Ep 260] Avg Reward: 2.504  | Actions: [np.float64(0.003), np.float64(0.61)]
[Ep 270] Avg Reward: 2.517  | Actions: [np.float64(0.001), np.float64(0.612)]
[Ep 280] Avg Reward: 2.521  | Actions: [np.float64(0.002), np.float64(0.611)]
[Ep 290] Avg Reward: 2.531  | Actions: [np.float64(0.001), np.float64(0.613)]
[Ep 300] Avg Reward: 2.524  | Actions: [np.float64(0.001), np.float64(0.608)]
[Ep 310] Avg Reward: 2.534  | Actions: [np.float64(0.0), np.float64(0.612)]
[Ep 320] Avg Reward: 2.511  | Actions: [np.float64(0.0), np.float64(0.608)]
[Ep 330] Avg Reward: 2.513  | Actions: [np.float64(0.0), np.float64(0.61)]
[Ep 340] Avg Reward: 2.533  | Actions: [np.float64(-0.002), np.float64(0.606)]
[Ep 350] Avg Reward: 2.543  | Actions: [np.float64(-0.0), np.float64(0.608)]
[Ep 370] Avg Reward: 2.543  | Actions: [np.float64(-0.001), np.float64(0.609)]
[Ep 380] Avg Reward: 2.540  | Actions: [np.float64(0.001), np.float64(0.605)]
[Ep 390] Avg Reward: 2.553  | Actions: [np.float64(-0.0), np.float64(0.608)]
[Ep 400] Avg Reward: 2.547  | Actions: [np.float64(-0.002), np.float64(0.608)]
[Ep 410] Avg Reward: 2.544  | Actions: [np.float64(-0.001), np.float64(0.611)]
[Ep 420] Avg Reward: 2.554  | Actions: [np.float64(-0.001), np.float64(0.607)]
[Ep 430] Avg Reward: 2.560  | Actions: [np.float64(-0.001), np.float64(0.608)]
[Ep 440] Avg Reward: 2.558  | Actions: [np.float64(-0.0), np.float64(0.605)]
[Ep 450] Avg Reward: 2.546  | Actions: [np.float64(-0.0), np.float64(0.603)]
[Ep 460] Avg Reward: 2.560  | Actions: [np.float64(0.001), np.float64(0.609)]
[Ep 470] Avg Reward: 2.560  | Actions: [np.float64(0.0), np.float64(0.608)]
[Ep 480] Avg Reward: 2.557  | Actions: [np.float64(-0.0), np.float64(0.596)]
[Ep 490] Avg Reward: 2.563  | Actions: [np.float64(0.001), np.float64(0.602)]
PS D:\Develop\Codes\UAV\self-ju\train\I_GitRemote\UAV-combat> & D:/Develop/anaconda3/envs/SE/python.exe d:/Develop/Codes/UAV/self-ju/train/I_GitRemote/UAV-combat/dense_test.py
[Ep 0] Avg Reward: 1.677  | Actions: [np.float64(0.16), np.float64(0.527)]
d:\Develop\Codes\UAV\self-ju\train\I_GitRemote\UAV-combat\dense_test.py:104: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\utils\tensor_new.cpp:257.)
  torch.FloatTensor(obs).to(device),      # [B, N, obs_dim]
[Ep 10] Avg Reward: 0.717  | Actions: [np.float64(-0.996), np.float64(0.063)]
[Ep 20] Avg Reward: 1.642  | Actions: [np.float64(-0.187), np.float64(0.368)]
[Ep 30] Avg Reward: 1.712  | Actions: [np.float64(-0.689), np.float64(-0.118)]
[Ep 40] Avg Reward: 1.683  | Actions: [np.float64(-0.492), np.float64(0.101)]
[Ep 50] Avg Reward: 1.680  | Actions: [np.float64(-0.427), np.float64(0.182)]
[Ep 60] Avg Reward: 1.693  | Actions: [np.float64(-0.682), np.float64(-0.079)]
[Ep 70] Avg Reward: 1.743  | Actions: [np.float64(-0.788), np.float64(-0.174)]
[Ep 80] Avg Reward: 1.769  | Actions: [np.float64(-0.728), np.float64(-0.125)]
[Ep 90] Avg Reward: 1.898  | Actions: [np.float64(-0.561), np.float64(0.052)]
[Ep 100] Avg Reward: 1.842  | Actions: [np.float64(-0.557), np.float64(0.051)]
[Ep 110] Avg Reward: 1.892  | Actions: [np.float64(-0.425), np.float64(0.183)]
[Ep 120] Avg Reward: 1.921  | Actions: [np.float64(-0.576), np.float64(0.022)]
[Ep 130] Avg Reward: 1.992  | Actions: [np.float64(-0.652), np.float64(-0.048)]
[Ep 140] Avg Reward: 2.016  | Actions: [np.float64(-0.512), np.float64(0.092)]
[Ep 150] Avg Reward: 2.099  | Actions: [np.float64(-0.64), np.float64(-0.03)]
[Ep 160] Avg Reward: 2.129  | Actions: [np.float64(-0.563), np.float64(0.042)]
[Ep 170] Avg Reward: 2.200  | Actions: [np.float64(-0.647), np.float64(-0.038)]
[Ep 180] Avg Reward: 2.285  | Actions: [np.float64(-0.567), np.float64(0.037)]
[Ep 190] Avg Reward: 2.383  | Actions: [np.float64(-0.597), np.float64(0.008)]
[Ep 200] Avg Reward: 2.523  | Actions: [np.float64(-0.603), np.float64(0.0)]
[Ep 210] Avg Reward: 2.520  | Actions: [np.float64(-0.603), np.float64(-0.001)]
[Ep 220] Avg Reward: 2.536  | Actions: [np.float64(-0.604), np.float64(0.002)]
[Ep 230] Avg Reward: 2.533  | Actions: [np.float64(-0.601), np.float64(0.002)]
[Ep 240] Avg Reward: 2.535  | Actions: [np.float64(-0.601), np.float64(0.003)]
[Ep 250] Avg Reward: 2.534  | Actions: [np.float64(-0.607), np.float64(0.002)]
[Ep 260] Avg Reward: 2.534  | Actions: [np.float64(-0.604), np.float64(0.001)]
[Ep 270] Avg Reward: 2.535  | Actions: [np.float64(-0.603), np.float64(0.001)]
[Ep 280] Avg Reward: 2.530  | Actions: [np.float64(-0.603), np.float64(-0.001)]
[Ep 290] Avg Reward: 2.538  | Actions: [np.float64(-0.604), np.float64(0.001)]
[Ep 300] Avg Reward: 2.525  | Actions: [np.float64(-0.6), np.float64(-0.001)]
[Ep 310] Avg Reward: 2.537  | Actions: [np.float64(-0.604), np.float64(0.0)]
[Ep 320] Avg Reward: 2.529  | Actions: [np.float64(-0.604), np.float64(0.0)]
[Ep 330] Avg Reward: 2.538  | Actions: [np.float64(-0.605), np.float64(-0.001)]
[Ep 340] Avg Reward: 2.536  | Actions: [np.float64(-0.601), np.float64(-0.0)]
[Ep 350] Avg Reward: 2.540  | Actions: [np.float64(-0.602), np.float64(0.001)]
[Ep 370] Avg Reward: 2.539  | Actions: [np.float64(-0.603), np.float64(0.001)]
[Ep 380] Avg Reward: 2.546  | Actions: [np.float64(-0.603), np.float64(-0.0)]
[Ep 390] Avg Reward: 2.543  | Actions: [np.float64(-0.603), np.float64(0.001)]
[Ep 400] Avg Reward: 2.555  | Actions: [np.float64(-0.605), np.float64(-0.0)]
[Ep 410] Avg Reward: 2.555  | Actions: [np.float64(-0.605), np.float64(0.0)]
[Ep 420] Avg Reward: 2.557  | Actions: [np.float64(-0.601), np.float64(0.001)]
[Ep 430] Avg Reward: 2.558  | Actions: [np.float64(-0.603), np.float64(-0.001)]
[Ep 440] Avg Reward: 2.556  | Actions: [np.float64(-0.602), np.float64(0.0)]
[Ep 450] Avg Reward: 2.560  | Actions: [np.float64(-0.6), np.float64(0.0)]
[Ep 460] Avg Reward: 2.549  | Actions: [np.float64(-0.602), np.float64(-0.0)]
[Ep 470] Avg Reward: 2.556  | Actions: [np.float64(-0.602), np.float64(-0.001)]
[Ep 480] Avg Reward: 2.561  | Actions: [np.float64(-0.602), np.float64(-0.0)]
[Ep 490] Avg Reward: 2.556  | Actions: [np.float64(-0.603), np.float64(0.0)]
PS D:\Develop\Codes\UAV\self-ju\train\I_GitRemote\UAV-combat> & D:/Develop/anaconda3/envs/SE/python.exe d:/Develop/Codes/UAV/self-ju/train/I_GitRemote/UAV-combat/dense_test.py
[Ep 0] Avg Reward: 1.677  | Actions: [np.float64(0.16), np.float64(0.527)]
d:\Develop\Codes\UAV\self-ju\train\I_GitRemote\UAV-combat\dense_test.py:104: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\utils\tensor_new.cpp:257.)
  torch.FloatTensor(obs).to(device),      # [B, N, obs_dim]
[Ep 10] Avg Reward: 0.899  | Actions: [np.float64(-1.0), np.float64(-1.0)]
[Ep 20] Avg Reward: 1.685  | Actions: [np.float64(0.092), np.float64(0.568)]
[Ep 30] Avg Reward: 1.697  | Actions: [np.float64(-0.22), np.float64(0.353)]
[Ep 40] Avg Reward: 1.675  | Actions: [np.float64(0.073), np.float64(0.676)]
[Ep 50] Avg Reward: 1.690  | Actions: [np.float64(0.176), np.float64(0.777)]
[Ep 60] Avg Reward: 1.703  | Actions: [np.float64(-0.085), np.float64(0.522)]
[Ep 70] Avg Reward: 1.750  | Actions: [np.float64(-0.175), np.float64(0.426)]
[Ep 80] Avg Reward: 1.778  | Actions: [np.float64(-0.129), np.float64(0.484)]
[Ep 90] Avg Reward: 1.910  | Actions: [np.float64(0.038), np.float64(0.638)]
[Ep 100] Avg Reward: 1.833  | Actions: [np.float64(0.044), np.float64(0.643)]
[Ep 110] Avg Reward: 1.886  | Actions: [np.float64(0.177), np.float64(0.788)]
[Ep 120] Avg Reward: 1.923  | Actions: [np.float64(0.028), np.float64(0.632)]
[Ep 130] Avg Reward: 1.992  | Actions: [np.float64(-0.049), np.float64(0.55)]
[Ep 140] Avg Reward: 2.018  | Actions: [np.float64(0.093), np.float64(0.687)]
[Ep 150] Avg Reward: 2.109  | Actions: [np.float64(-0.032), np.float64(0.568)]
[Ep 160] Avg Reward: 2.127  | Actions: [np.float64(0.041), np.float64(0.639)]
[Ep 170] Avg Reward: 2.206  | Actions: [np.float64(-0.039), np.float64(0.566)]
[Ep 180] Avg Reward: 2.284  | Actions: [np.float64(0.036), np.float64(0.64)]
[Ep 190] Avg Reward: 2.383  | Actions: [np.float64(0.011), np.float64(0.612)]
[Ep 200] Avg Reward: 2.523  | Actions: [np.float64(0.0), np.float64(0.601)]
[Ep 210] Avg Reward: 2.527  | Actions: [np.float64(-0.004), np.float64(0.594)]
[Ep 220] Avg Reward: 2.519  | Actions: [np.float64(-0.007), np.float64(0.583)]
[Ep 230] Avg Reward: 2.527  | Actions: [np.float64(-0.005), np.float64(0.586)]
[Ep 240] Avg Reward: 2.534  | Actions: [np.float64(-0.003), np.float64(0.593)]
[Ep 250] Avg Reward: 2.533  | Actions: [np.float64(-0.011), np.float64(0.581)]
[Ep 260] Avg Reward: 2.520  | Actions: [np.float64(-0.0), np.float64(0.597)]
[Ep 270] Avg Reward: 2.527  | Actions: [np.float64(-0.001), np.float64(0.595)]
[Ep 280] Avg Reward: 2.533  | Actions: [np.float64(-0.0), np.float64(0.594)]
[Ep 290] Avg Reward: 2.538  | Actions: [np.float64(-0.001), np.float64(0.595)]
[Ep 300] Avg Reward: 2.552  | Actions: [np.float64(0.0), np.float64(0.598)]
[Ep 310] Avg Reward: 2.547  | Actions: [np.float64(0.0), np.float64(0.596)]
[Ep 320] Avg Reward: 2.548  | Actions: [np.float64(0.001), np.float64(0.598)]
[Ep 330] Avg Reward: 2.555  | Actions: [np.float64(0.0), np.float64(0.594)]
[Ep 340] Avg Reward: 2.546  | Actions: [np.float64(0.0), np.float64(0.596)]
[Ep 350] Avg Reward: 2.556  | Actions: [np.float64(0.002), np.float64(0.598)]
[Ep 360] Avg Reward: 2.545  | Actions: [np.float64(-0.0), np.float64(0.596)]
[Ep 370] Avg Reward: 2.540  | Actions: [np.float64(0.0), np.float64(0.597)]
[Ep 380] Avg Reward: 2.550  | Actions: [np.float64(0.0), np.float64(0.594)]
[Ep 390] Avg Reward: 2.546  | Actions: [np.float64(0.0), np.float64(0.592)]
[Ep 400] Avg Reward: 2.545  | Actions: [np.float64(0.001), np.float64(0.588)]
[Ep 410] Avg Reward: 2.557  | Actions: [np.float64(0.0), np.float64(0.594)]
[Ep 420] Avg Reward: 2.560  | Actions: [np.float64(0.001), np.float64(0.591)]
[Ep 430] Avg Reward: 2.562  | Actions: [np.float64(-0.0), np.float64(0.591)]
[Ep 440] Avg Reward: 2.555  | Actions: [np.float64(0.001), np.float64(0.594)]
[Ep 450] Avg Reward: 2.558  | Actions: [np.float64(0.0), np.float64(0.591)]
[Ep 460] Avg Reward: 2.553  | Actions: [np.float64(0.001), np.float64(0.591)]
[Ep 470] Avg Reward: 2.562  | Actions: [np.float64(-0.0), np.float64(0.588)]
[Ep 480] Avg Reward: 2.565  | Actions: [np.float64(0.0), np.float64(0.588)]
[Ep 490] Avg Reward: 2.564  | Actions: [np.float64(-0.0), np.float64(0.586)]
PS D:\Develop\Codes\UAV\self-ju\train\I_GitRemote\UAV-combat> 
"""


