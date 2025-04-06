# from env import Drone
import numpy as np
import math

MAX_SPEED = 8
MAX_ANGLE_SPEED = np.pi / 16
MAX_ACCELERATE = 4
MAX_ANGLE_ACCE = np.pi / 32

MAP_SIZE_0 = 750
MAP_SIZE_1 = 750



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
    


class DroneRewardSecond:
    def __init__(self, drones: list, actions: list):
        self.drones = drones
        self.actions = actions
        self.last_positions = {drone.id:(drone.x, drone.y) for drone in drones}
        
    def _get_enemies(self, drone):
        return [d for d in self.drones if d.teamcode != drone.teamcode and d.alive]
    
    def _get_allies(self, drone):
        return [d for d in self.drones if d.teamcode == drone.teamcode and d.id != drone.id and d.alive]

    def update_and_return_deprive(self):
        rewards = np.zeros(len(self.drones))
        
        for idx, drone in enumerate(self.drones):
            action = self.actions[idx]
            a, phi, shoot = action

            # if abs(a) <= 0.1:
            #     rewards[idx] += 0.1

            # if abs(phi) <= 0.1:
            #     rewards[idx] += 0.1

            if shoot < 0:
                rewards[idx] += 1

            # flag_small_angle_speed = bool(abs(drone.w) <= 0.25 * MAX_ANGLE_SPEED)
            # flag_angle_speed_decrease = bool(abs(phi) <= 0.25)

            # flag_proper_speed = bool(drone.v>0.25*MAX_SPEED and drone.v>0.5*MAX_SPEED)

            # if flag_small_angle_speed or flag_angle_speed_decrease:
            #     rewards[idx] += 0.2

            # if flag_proper_speed:
            #     rewards[idx] += 0.2

        return rewards
    



    def update_and_return(self):
        rewards = np.zeros(len(self.drones))
        
        for idx, drone in enumerate(self.drones):
            if not drone.alive:
                rewards[idx] -= 0.5  # 持续死亡惩罚
                continue
                
            enemies = self._get_enemies(drone)
            allies = self._get_allies(drone)
            action = self.actions[idx]
            
            # ===== 基础奖励 =====
            rewards[idx] += 0.1  # 生存奖励
            
            # ===== 攻击系统 =====
            # 成功开火奖励（在step()中已实现）
            # 冷却期闲置惩罚
            if drone.fire_cooltime <= 0 and action[2] < 0:
                rewards[idx] -= 0.2
                
            # ===== 移动策略 =====
            # 推进效率（鼓励合理加速）
            rewards[idx] += 0.02 * (1 - abs(action[0])) 
            
            # # 转向效率（鼓励平滑转向）
            rewards[idx] += 0.01 * (1 - abs(action[1]))

            # 角速度不太大奖励（自己想的）
            # if abs(drone.w) > 0.25*MAX_ANGLE_SPEED:
            #     rewards[idx] -= 0.2
            # if abs(drone.w) > 0.5*MAX_ANGLE_SPEED:
            #     rewards[idx] -= 0.2
            
            # ===== 战术定位 =====
            if enemies:
                # 获取最近敌人
                nearest = min(enemies, key=lambda e: (e.x-drone.x)**2 + (e.y-drone.y)**2)
                dist = math.hypot(nearest.x-drone.x, nearest.y-drone.y)
                
                # 动态距离奖励（最优攻击距离区间）
                if 100 < dist < 400:
                    rewards[idx] += 0.05 * (1 - abs(dist-250)/150)
                else:
                    rewards[idx] -= 0.03
                    
                # 朝向奖励（航向角与目标夹角）
                target_angle = math.atan2(nearest.y-drone.y, nearest.x-drone.x)
                angle_diff = abs((drone.orientation - target_angle) % (2*math.pi))
                rewards[idx] += 0.1 * (1 - angle_diff/math.pi)
                
                # 移动趋势奖励（与上次位置变化）
                dx = drone.x - self.last_positions[drone.id][0]
                dy = drone.y - self.last_positions[drone.id][1]
                move_vec = math.atan2(dy, dx)
                move_diff = abs((move_vec - target_angle) % (2*math.pi))
                rewards[idx] += 0.05 * (1 - move_diff/math.pi)
            
            # ===== 团队协作 =====
            # 集群密度奖励（鼓励保持队形）
            if allies:
                avg_dist = sum(math.hypot(a.x-drone.x, a.y-drone.y) for a in allies)/len(allies)
                rewards[idx] += 0.03 * (1 - avg_dist/500) if avg_dist < 500 else -0.02
                
            # 支援奖励（队友攻击时）
            # for ally in allies:
            #     if ally.fire_cooltime == MAX_FIRE_COOLDOWN-1:  # 检测到队友刚开火
            #         rewards[idx] += 0.1 * (1 - math.hypot(ally.x-drone.x, ally.y-drone.y)/800)

            # ===== 边界惩罚 =====
            if (drone.x < 50 or drone.x > MAP_SIZE_0-50 or 
                drone.y < 50 or drone.y > MAP_SIZE_1-50):
                rewards[idx] -= 0.2
                
            # 更新位置记录
            self.last_positions[drone.id] = (drone.x, drone.y)
            
        # ===== 团队胜负奖励 ===== 
        # red_alive = sum(1 for d in self.drones if d.team=='red' and d.alive)
        # blue_alive = sum(1 for d in self.drones if d.team=='blue' and d.alive)
        
        # for idx, drone in enumerate(self.drones):
        #     if drone.team == 'red' and blue_alive == 0:
        #         rewards[idx] += 20
        #     elif drone.team == 'blue' and red_alive == 0:
        #         rewards[idx] += 20
                
        return rewards