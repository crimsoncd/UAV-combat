
from config import *

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal

import itertools

def euclidean_distance(item_a, item_b):
    return np.linalg.norm(np.array(item_a) - np.array(item_b))

def optimal_matching(A, B):
    num_a = len(A)
    num_b = len(B)

    # 构建代价矩阵
    cost_matrix = np.zeros((num_a, num_b))

    for i, a in enumerate(A):
        for j, b in enumerate(B):
            cost_matrix[i, j] = euclidean_distance(a, b)

    # 匈牙利算法求解最优匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 如果B的数量大于5，则只保留前5个最小代价的
    if num_b > 5:
        sorted_indices = np.argsort(cost_matrix[row_ind, col_ind])
        row_ind = row_ind[sorted_indices][:5]
        col_ind = col_ind[sorted_indices][:5]

    # 返回匹配结果
    matches = [(row, col) for row, col in zip(row_ind, col_ind)]

    # 如果有未匹配的A元素，补上None
    matched_indices = set(row_ind)
    for i, a in enumerate(A):
        if i not in matched_indices:
            matches.append((i, None))

    return matches






class EnemyDataPack():

    def __init__(self):
        self.packs = []

    def __len__(self):
        return len(self.packs)

    def Unpack_and_Store(self, input_obs):
        # input_obs结构：List of Lists
        # 每一个List表示一个obs, obs=[info_self(5), info_enemy(5)*n]
        # info_enemy(5): 1/0, x, y, v, orientation
        pack = []
        # Unpack to fragments
        enemies_items = []
        for obs in input_obs:
            num_enemy = len(obs[5:]) // 5
            for n in range(1, num_enemy+1):
                if obs[n*5]>0:
                    enemy_item = obs[n*5+1:n*5+5]
                    if enemy_item not in enemies_items:
                        enemies_items.append(enemy_item)
        # Regenerate Data
        # Every pack is a list of (x, y, v, o) (Standardized)
        for item in enemies_items:
            d = (item[0], 
                 item[1], 
                 item[2], 
                 item[3])
            pack.append(d)
        # Store
        self.packs.append(pack)

    def Pop(self):
        return self.packs[-1]




class EnemySpyer:

    def __init__(self, process_noise=1.0, measurement_noise=1.5):
        # Bayesian Parameter
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        # Sport Parameter
        self.position_belief = None
        self.last_v = None
        self.last_o = None
    
    def initialize(self, packed_info, initial_uncertainty=(5.0, 5.0)):
        # packed_info: (x, y, v, o)
        # Warning: postion and other paras in this class is in original range
        # rather than standardized, eg: 0 <= x <= MAP_SIZE_0  (instead of 0~1)
        x, y, self.last_v, self.last_o = packed_info
        mean = np.array([x, y])
        cov = np.diag(initial_uncertainty)
        self.position_belief = multivariate_normal(mean=mean, cov=cov)
        return [1.0, x / MAP_SIZE_0, y / MAP_SIZE_1, self.last_v / MAX_SPEED, self.last_o / np.pi]

    def predict(self):
        if self.position_belief is None:
            return [0, 0, 0, 0, 0]
        vx, vy = self.last_v*np.cos(self.last_o), self.last_v*np.sin(self.last_o)
        mean = self.position_belief.mean + np.array([vx, vy])
        cov = self.position_belief.cov + np.eye(2) * self.process_noise
        self.position_belief = multivariate_normal(mean=mean, cov=cov)
        return [1.0, mean[0] / MAP_SIZE_0, mean[1] / MAP_SIZE_1, self.last_v / MAX_SPEED, self.last_o / np.pi]
    
    def update(self, packed_info):
        if self.position_belief is None:
            return self.initialize(packed_info)

        x, y, self.last_v, self.last_o = packed_info

        # likelihood = multivariate_normal(mean=[x, y], cov=np.eye(2) * self.measurement_noise)
        prior_mean = self.position_belief.mean
        prior_cov = self.position_belief.cov
        kalman_gain = np.dot(prior_cov, np.linalg.inv(prior_cov + np.eye(2) * self.measurement_noise))
        updated_mean = prior_mean + np.dot(kalman_gain, ([x, y] - prior_mean))
        updated_cov = (np.eye(2) - kalman_gain).dot(prior_cov)
        self.position_belief = multivariate_normal(mean=updated_mean, cov=updated_cov)

        return [1.0, x / MAP_SIZE_0, y / MAP_SIZE_1, self.last_v / MAX_SPEED, self.last_o / np.pi]
    
    def get_data(self):
        if self.position_belief is None:
            return [0, 0, 0, 0, 0]
        x, y = self.position_belief.mean
        return [1.0, x / MAP_SIZE_0, y / MAP_SIZE_1, self.last_v / MAX_SPEED, self.last_o / np.pi]
    
    def get_standard_data(self):
        if self.position_belief is None:
            return [0, 0, 0, 0]
        x, y = self.position_belief.mean
        return [x / MAP_SIZE_0, y / MAP_SIZE_1, self.last_v / MAX_SPEED, self.last_o / np.pi]



class BlackBox():

    def __init__(self, num_enemy=5):
        self.n = num_enemy
        self.Pack = EnemyDataPack()
        self.Spyers = [EnemySpyer() for _ in range(num_enemy)]

    def reset(self):
        self.Pack = EnemyDataPack()
        self.Spyers = [EnemySpyer() for _ in range(self.n)]

    def update(self, input_obs):

        self.Pack.Unpack_and_Store(input_obs)
        packs = self.Pack.Pop()

        # Every pack is a list of (x, y, v, o) (Standardized) or an empty list []
        if len(packs) == 0:
            predicts = [self.Spyers[i].predict() for i in range(self.n)]
            return list(itertools.chain.from_iterable(predicts))
        
        # Have some obs: update if there is an obs, else predict
        r_enemies = [self.Spyers[i].get_standard_data() for i in range(self.n)]
        r_obs = packs
        matches = optimal_matching(r_enemies, r_obs)
        if len(matches) != self.n:
            raise ValueError("Match outcome is not consistent with number of enemies")

        results = [None] * self.n
        for match in matches:
            # print("matching:", match)
            enemy_idx, pack_idx = match
            if pack_idx is None:
                enemy_idx = int(match[0])
                results[enemy_idx] = self.Spyers[enemy_idx].predict()
            else:
                enemy_idx, pack_idx = int(match[0]), int(match[1])
                results[enemy_idx] = self.Spyers[enemy_idx].update(packs[pack_idx])
        
        # print(results)
        return list(itertools.chain.from_iterable(results))

















if __name__=="__main__":
    obs_i = [9, 9, 9, 9, 9, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 3, 3, 5, 4]
    obs_j = [9, 9, 9, 9, 9, 1, 3, 3, 5, 4, 1, 8, 8, 9, 0, 0, 0, 0, 0, 0]
    obs_k = [9, 9, 9, 9, 9, 1, 3, 3, 5, 4, 1, 8, 8, 9, 0, 0, 0, 0, 0, 0]
    obs_all = [obs_i, obs_j, obs_k]
    
    BB = BlackBox(5)
    result1 = BB.update(obs_all)
    print(len(result1), result1)

    obs_i = [9, 9, 9, 9, 9, 1, 2, 3, 4, 6, 1, 2, 3, 4, 6, 1, 3, 3, 5, 4]
    obs_j = [9, 9, 9, 9, 9, 1, 3, 3, 5, 4, 1, 8, 8, 10, 0, 0, 0, 0, 0, 0]
    obs_k = [9, 9, 9, 9, 9, 1, 3, 3, 5, 4, 1, 8, 8, 10, 0, 0, 0, 0, 0, 0]
    obs_all = [obs_i, obs_j, obs_k]
    result2 = BB.update(obs_all)
    print(len(result2), result2)

    # [2, 3, 4, 5, 3, 3, 5, 4, 8, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # [2, 3, 4, 6, 3, 3, 5, 4, 8, 8, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Good!


