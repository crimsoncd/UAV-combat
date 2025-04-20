# from env import Drone
import numpy as np
import math

MAX_SPEED = 8
MAX_ANGLE_SPEED = np.pi / 16 
MAX_ACCELERATE = 4
MAX_ANGLE_ACCE = np.pi / 32

MAP_SIZE_0 = 750
MAP_SIZE_1 = 750

FIRE_RANGE = 50


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

def control_strategy_A(self_drone, enemy_x, enemy_y):
    """生成控制指令以接近敌机并攻击"""
    dx = enemy_x - self_drone.x
    dy = enemy_y - self_drone.y
    distance = np.hypot(dx, dy)
    
    # 计算目标方向
    target_dir = np.arctan2(dy, dx)
    # 计算方向误差（归一化到[-π, π]）
    delta_theta = target_dir - self_drone.orientation
    delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi
    
    # 角加速度控制：比例控制
    Kp_phi = 0.1  # 比例系数，调整转向灵敏度
    phi = Kp_phi * delta_theta
    phi = np.clip(phi, -MAX_ANGLE_ACCE, MAX_ANGLE_ACCE)
    
    # 加速度控制：方向正确时加速，否则减速
    if abs(delta_theta) < np.pi/6:  # 误差小于30度时加速
        a = MAX_ACCELERATE
    else:
        a = -MAX_ACCELERATE * 0.5  # 反向减速以方便转向
    
    # 发射指令：在射程内且冷却完毕
    fire = -1
    if distance < FIRE_RANGE and self_drone.fire_cooltime <= 0:
        fire = 1
    
    return a, phi, fire

def control_strategy_B(drone, enemy_x, enemy_y):
    """
    控制己方无人机朝向敌机，并在合适时机进行攻击
    返回加速度a, 角加速度phi, 是否发射fire(boolean)
    """

    # 1. 计算当前位置与目标之间的向量
    dx = enemy_x - drone.x
    dy = enemy_y - drone.y
    target_angle = np.arctan2(dy, dx)  # 敌机相对角度

    # 2. 当前机头朝向
    current_angle = drone.orientation

    # 3. 角度误差归一化到 [-pi, pi]
    angle_diff = ((target_angle - current_angle + np.pi) % (2 * np.pi)) - np.pi

    # 4. 控制角加速度 phi，让机头慢慢对准目标
    K_phi = 0.1  # 控制增益，可以调整
    phi = np.clip(K_phi * angle_diff - drone.w, -MAX_ANGLE_ACCE, MAX_ANGLE_ACCE)

    # 5. 控制加速度 a，简单策略是机头方向与目标方向越接近，a越大
    if abs(angle_diff) < np.pi / 6:  # 如果偏差在30°以内
        a = MAX_ACCELERATE
    else:
        a = 0  # 停止加速，先转头

    # 6. 判断是否进入攻击范围
    distance = np.sqrt(dx**2 + dy**2)
    fire =  1 if distance < FIRE_RANGE and abs(angle_diff) < np.pi / 12 else -1 # 距离近且对准敌人

    return a, phi, fire

def control_strategy_C(drone, enemy_x, enemy_y):
    # 参数配置（可以调优）
    K_phi = 0.2
    K_brake = 1.0
    K_acc = 0.05
    ANGLE_THRESHOLD = np.pi / 8  # ~22.5度以内才能加速
    CLOSE_RANGE = 5.0
    MAX_RANGE = 30.0

    dx = enemy_x - drone.x
    dy = enemy_y - drone.y
    distance = np.hypot(dx, dy)
    
    # 敌人的相对角度
    target_angle = np.arctan2(dy, dx)
    current_angle = drone.orientation

    # 角度误差归一化 [-pi, pi]
    angle_diff = ((target_angle - current_angle + np.pi) % (2 * np.pi)) - np.pi

    # ===== 角加速度控制 =====
    # 控制目标角速度（希望的转速），目标角度越大希望转得越快
    target_w = K_phi * angle_diff
    # 加入阻尼（刹车项）避免一直转圈
    phi = np.clip(target_w - K_brake * drone.w, -MAX_ANGLE_ACCE, MAX_ANGLE_ACCE)

    # ===== 加速度控制 =====
    # 若角度误差较小才加速
    if abs(angle_diff) < ANGLE_THRESHOLD:
        # 距离越远，加速度越大；近处则放缓
        acc_factor = min(distance / MAX_RANGE, 1.0)
        a = K_acc * acc_factor
    else:
        a = 0  # 角度没对上，不加速

    # ===== 发射控制 =====
    fire = 1 if distance < FIRE_RANGE and abs(angle_diff) < np.pi / 12 else -1 # 距离近且对准敌人

    return a, phi, fire






class GPTReward:
    def __init__(self, drones, actions, team_win_bonus=10):
        self.drones = drones
        self.actions = actions
        self.num_agents = len(drones)
        self.rewards = np.zeros(self.num_agents)
        self.rewards_on_type = [0, 0, 0]

    def _get_enemies(self, drone):
        return [d for d in self.drones if d.teamcode != drone.teamcode and d.alive]

    def update_and_return(self):
        for idx, drone in enumerate(self.drones):

            if not drone.alive:
                # self.rewards[idx] -= 1  # 死亡惩罚
                continue

            action = self.actions[idx]
            enemies = self._get_enemies(drone)

            # 生存奖励
            self.rewards[idx] += 0.1
            self.rewards_on_type[0] += 0.1

            # 面朝敌人奖励（航向角误差）
            if enemies:
                nearest = min(enemies, key=lambda e: (e.x - drone.x) ** 2 + (e.y - drone.y) ** 2)
                dx, dy = nearest.x - drone.x, nearest.y - drone.y
                target_angle = math.atan2(dy, dx)
                angle_diff = abs((drone.orientation - target_angle + math.pi) % (2 * math.pi) - math.pi)
                angle_reward = 0.2 * (1 - angle_diff / math.pi)
                self.rewards[idx] += angle_reward
                self.rewards_on_type[1] += angle_reward

            # 发射判断
            if drone.fire_cooltime <= 0 and action[2] > 0:
                self.rewards[idx] += 0.5  # 鼓励主动攻击
                self.rewards_on_type[1] += 0.5

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