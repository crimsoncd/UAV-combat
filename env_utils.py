# from env import Drone
import numpy as np
import math

# Small utils
def rewind(input_num, lower_bound, upper_bound, maintain_int=True):
    # This function is used to cut the number in a range
    # For example: 6 in 0~7 is 6, 9 in 0~7 is 1, -3 in 0~7 is 5
    if maintain_int:
        i, l, u = np.floor(input_num), np.floor(lower_bound), np.floor(upper_bound)
    else:
        i, l, u = input_num, lower_bound, upper_bound
    if i < l:
        return rewind(i+u-l, l, u)
    elif i > u:
        return rewind(i-u+l, l, u)
    else:
        return i

def nearest_direction(self_x, self_y, target_x, target_y):
    dx = target_x - self_x
    dy = target_y - self_y
    if dx == 0 and dy == 0:
        return 0
    # 计算角度并转换为0~360度
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad) % 360.0
    # 计算最近的方向
    direction = int((angle_deg + 22.5) // 45) % 8
    return direction

def noise_mask(array, mask_prob=0.25, mask_range=0.1):
    n_array = []
    for a in array:
        if np.random.random() <= mask_prob:
            a += a*np.random.uniform(-mask_range*a, mask_range*a)
        n_array.append(a)
    return n_array

# def induce_step_from_state(state):
#     self_drone = self.drones[idx]
#     target_drone = np.random.choice([enemy for enemy in self.drones if enemy.teamcode!=self_drone.teamcode])
#     distance = (self_drone.x - target_drone.x)**2 + (self_drone.y - target_drone.y)**2
#     if distance <= 40000 and self_drone.cool_time<=0:
#         return 9
#     else:
#         return nearest_direction(self_drone.x, self_drone.y, target_drone.x, target_drone.y) + 1







class DroneReward():

    def __init__(self, drones: list, actions: list, map_size=(900, 900), test_mode=False):
        self.test_mode = test_mode
        # Store values
        self.drones = drones
        self.num_drones = len(drones)
        self.actions = actions
        self.map_size = map_size
        self.rewards = np.zeros(self.num_drones)
        # Parameters
        self.alive_reward = 0.05
        self.approach_reward_formula = lambda x: 0.7-(x/500000)
        self.position_reward_base = 0.05

    # 存活奖励
    def single_alive_reward(self, idx):
        if self.drones[idx].alive:
            return self.alive_reward
        return 0
    
    def update_alive_reward(self):
        for i in range(self.num_drones):
            self.rewards[i] += self.single_alive_reward(i)

    # 接近敌机奖励，越接近分数越高
    def single_approach_reward(self, idx: int):
        if self.drones[idx].alive==False:
            return 0
        # Nearest enemy
        nearest_distance = 999999
        for i in range(self.num_drones):
            if self.drones[i].team != self.drones[idx].team:
                delta_x, delta_y = abs(self.drones[i].x-self.drones[idx].x), abs(self.drones[i].y-self.drones[idx].y)
                distance_square = delta_x**2 + delta_y**2
                if distance_square <= nearest_distance:
                    nearest_distance = distance_square
        # Calculate reward
        # print("Get nearest distance", nearest_distance)    100k unit
        # distance_log = np.log10(nearest_distance)
        return self.approach_reward_formula(nearest_distance)
    
    def update_approach_reward(self):
        for i in range(self.num_drones):
            self.rewards[i] += self.single_approach_reward(i)

    # 位置奖励，越靠近中心分数越高
    def single_position_reward(self, idx: int):
        if self.drones[idx].alive==False:
            return 0
        x, y = self.drones[idx].x, self.drones[idx].y
        x_max, y_max = self.map_size[0], self.map_size[1]
        flag_x = 1 if (x>=0.33*x_max and x<=0.67*x_max) else 0
        flag_y = 1 if (y>=0.33*y_max and y<=0.67*y_max) else 0
        return (flag_x + flag_y) * self.position_reward_base
    
    def update_position_reward(self):
        for i in range(self.num_drones):
            self.rewards[i] += self.single_position_reward(i)

    def _single_test_reward(self, idx: int):
        return 1 if self.drones[idx].orientation==0 else 0
    
    def _update_test_reward(self):
        for i in range(self.num_drones):
            self.rewards[i] += self._single_test_reward(i)

    # 统计全部并输出
    def update_and_return(self):
        if self.test_mode:
            self._update_test_reward()
        else:
            self.update_alive_reward()
            self.update_approach_reward()
            self.update_position_reward()
        # print("Calculated rewards:", self.rewards)
        return self.rewards