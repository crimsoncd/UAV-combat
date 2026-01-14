# from env import Drone
import numpy as np
import math

from config import *

ATTACK_ALPHA = np.pi / 2
ATTACK_R = 100

FIRE_RANGE = 50

def is_in_sector(point1, point2, orientation, alpha=ATTACK_ALPHA, r=ATTACK_R):

    # 计算两点之间的距离
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    distance_sq = dx**2 + dy**2
    
    # 首先检查是否在半径范围内
    if distance_sq > r**2:
        return False
    
    angle = math.atan2(dy, dx)
    if angle < 0:
        angle += 2 * math.pi
    angle_diff = abs(angle - orientation)
    angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
    
    return angle_diff <= alpha / 2



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


def control_strategy_Expert(drone, all_drones):
    """
    专家策略：带有协同目标识别 + 距离控制 + 回避边界 + 稳态机制的行为逻辑
    """
    # 参数配置
    K_phi = 0.2
    K_brake = 0.9
    K_acc = 0.05
    SAFE_MARGIN = 50

    enemies = [d for d in all_drones if d.teamcode != drone.teamcode and d.alive
               and (d.x - drone.x)**2 + (d.y - drone.y)**2 <= RADIUS**2]
    # allies = [d for d in all_drones if d.teamcode == drone.teamcode and d.id != drone.id and d.alive]

    if not enemies:
        # ===== 巡逻模式（无敌机可见时） =====
        center_x, center_y = MAP_SIZE_0 / 2, MAP_SIZE_1 / 2
        dx, dy = center_x - drone.x, center_y - drone.y
        distance_center = np.hypot(dx, dy)

        # 简单巡逻逻辑：围绕中心点转圈
        patrol_radius = 200
        target_angle = math.atan2(dy, dx) + math.pi/2  # 在中心点逆时针巡逻

        # 防止飞出地图：边缘修正
        if drone.x < SAFE_MARGIN or drone.x > MAP_SIZE_0 - SAFE_MARGIN or \
           drone.y < SAFE_MARGIN or drone.y > MAP_SIZE_1 - SAFE_MARGIN:
            target_angle = math.atan2(center_y - drone.y, center_x - drone.x)

        current_angle = drone.orientation
        angle_diff = ((target_angle - current_angle + np.pi) % (2 * np.pi)) - np.pi

        # 角加速度调整
        target_w = K_phi * angle_diff
        phi = np.clip(target_w - K_brake * drone.w, -MAX_ANGLE_ACCE, MAX_ANGLE_ACCE)

        # 加速度控制：适度巡航
        if abs(angle_diff) < np.pi/8:
            a = K_acc * 1.0
        else:
            a = 0

        return a, phi, -1  # 不开火

    # 优先选择最近的敌人 or 距离长机最近者（可扩展）
    nearest = min(enemies, key=lambda e: (e.x - drone.x)**2 + (e.y - drone.y)**2)
    dx, dy = nearest.x - drone.x, nearest.y - drone.y
    distance = np.hypot(dx, dy)
    target_angle = math.atan2(dy, dx)
    current_angle = drone.orientation
    angle_diff = ((target_angle - current_angle + np.pi) % (2 * np.pi)) - np.pi

    # ========== 动作策略控制 ==========

    # 控制角加速度（朝向敌人）
    target_w = K_phi * angle_diff
    phi = np.clip(target_w - K_brake * drone.w, -MAX_ANGLE_ACCE, MAX_ANGLE_ACCE)

    # 控制加速度：
    # 1. 对准敌人才推进
    # 2. 如果太靠近敌人则减速
    if abs(angle_diff) < np.pi / 6:
        if distance < 30:
            a = -MAX_ACCELERATE * 0.3  # 太近减速
        else:
            a = min(K_acc * (distance / 50), MAX_ACCELERATE)
    else:
        a = 0

    # 控制边界回避行为
    if drone.x < SAFE_MARGIN or drone.x > MAP_SIZE_0 - SAFE_MARGIN or \
       drone.y < SAFE_MARGIN or drone.y > MAP_SIZE_1 - SAFE_MARGIN:
        target_angle = math.atan2(MAP_SIZE_1 / 2 - drone.y, MAP_SIZE_0 / 2 - drone.x)
        angle_diff = ((target_angle - drone.orientation + np.pi) % (2 * np.pi)) - np.pi
        phi = np.clip(K_phi * angle_diff - K_brake * drone.w, -MAX_ANGLE_ACCE, MAX_ANGLE_ACCE)
        a = MAX_ACCELERATE * 0.2

    # 发射控制：靠近+对准
    fire = 1 if distance < FIRE_RANGE and abs(angle_diff) < np.pi / 12 and drone.fire_cooltime <= 0 else -1

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
    
    def update_and_return_sample(self):

        rewards_sample = np.zeros(self.num_agents)
        # print("Using test reward method...")

        for idx, drone in enumerate(self.drones):

            if not drone.alive:
                continue

            action = self.actions[idx]
            reward_scale = 1

            rewards_sample[idx] += reward_scale * (1 - abs(action[0]))
            rewards_sample[idx] += reward_scale * (1 - abs(action[1]))
            rewards_sample[idx] += reward_scale * (1 - abs(action[2]))

            rewards_sample[idx] -= reward_scale * 1.5

        return rewards_sample

    def update_and_return_real(self):
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
                if dx**2 + dy**2 >= 200**2:
                    break
                target_angle = math.atan2(dy, dx)
                angle_diff = abs((drone.orientation - target_angle) % (2 * math.pi) - math.pi)
                angle_reward = 0.2 * (1 - angle_diff / math.pi)
                self.rewards[idx] += angle_reward
                self.rewards_on_type[1] += angle_reward

            # 发射判断
            if drone.fire_cooltime <= 0 and action[2] > 0:
                self.rewards[idx] += 0.5  # 鼓励主动攻击
                self.rewards_on_type[1] += 0.5

        return self.rewards






    


class DroneRewardSecond:
    def __init__(self, drones: list, actions: list, debug=False):
        self.drones = drones
        self.actions = actions
        self.last_positions = {drone.id:(drone.x, drone.y) for drone in drones}
        self.debug = debug
        self.reward_breakdown_log = []
        
    def _get_enemies(self, drone):
        return [d for d in self.drones if d.teamcode != drone.teamcode and d.alive]
    
    def _get_allies(self, drone):
        return [d for d in self.drones if d.teamcode == drone.teamcode and d.id != drone.id and d.alive]




    def update_and_return(self):
        rewards = np.zeros(len(self.drones))

        for idx, drone in enumerate(self.drones):
            breakdown = {
                'id': drone.id,
                'team': drone.team,
                'alive': int(drone.alive),
                'reward_survive': 0.0,
                'reward_fire_idle': 0.0,
                'reward_push': 0.0,
                'reward_turn': 0.0,
                'reward_distance': 0.0,
                'reward_facing': 0.0,
                'reward_movement': 0.0,
                'reward_cohesion': 0.0,
                'reward_border': 0.0,
                'total': 0.0
            }

            if not drone.alive:
                rewards[idx] -= 0.5
                breakdown['total'] = -0.5
                if self.debug:
                    print(f"[Drone {idx}] dead → reward: -0.5")
                self.reward_breakdown_log.append(breakdown)
                continue

            enemies = [d for d in self.drones if d.teamcode != drone.teamcode and d.alive]
            allies = [d for d in self.drones if d.teamcode == drone.teamcode and d.id != drone.id and d.alive]
            action = self.actions[idx]

            # --- 基础奖励 ---
            breakdown['reward_survive'] = 0.1
            rewards[idx] += breakdown['reward_survive']

            if drone.fire_cooltime <= 0 and action[2] < 0:
                breakdown['reward_fire_idle'] = -0.2
                rewards[idx] += breakdown['reward_fire_idle']

            breakdown['reward_push'] = float(0.02 * (1 - abs(action[0])))
            breakdown['reward_turn'] = float(0.01 * (1 - abs(action[1])))
            rewards[idx] += breakdown['reward_push'] + breakdown['reward_turn']

            # --- 面对敌人奖励 ---
            if enemies:
                nearest = min(enemies, key=lambda e: (e.x-drone.x)**2 + (e.y-drone.y)**2)
                dist = math.hypot(nearest.x-drone.x, nearest.y-drone.y)
                dist = float(dist)
                if 100 < dist < 400:
                    breakdown['reward_distance'] = 0.05 * (1 - abs(dist - 250) / 150)
                else:
                    breakdown['reward_distance'] = -0.03
                rewards[idx] += breakdown['reward_distance']

                target_angle = math.atan2(nearest.y-drone.y, nearest.x-drone.x)
                angle_diff = abs((drone.orientation - target_angle) % (2 * math.pi))
                breakdown['reward_facing'] = float(0.1 * (1 - angle_diff / math.pi))
                rewards[idx] += breakdown['reward_facing']

                dx = drone.x - self.last_positions[drone.id][0]
                dy = drone.y - self.last_positions[drone.id][1]
                move_vec = math.atan2(dy, dx)
                move_diff = abs((move_vec - target_angle) % (2 * math.pi))
                breakdown['reward_movement'] = float(0.05 * (1 - move_diff / math.pi))
                rewards[idx] += breakdown['reward_movement']

            # --- 队形密度奖励 ---
            if allies:
                avg_dist = sum(math.hypot(a.x - drone.x, a.y - drone.y) for a in allies) / len(allies)
                breakdown['reward_cohesion'] = 0.03 * (1 - avg_dist / 500) if avg_dist < 500 else -0.02
                rewards[idx] += breakdown['reward_cohesion']

            # --- 边界惩罚 ---
            if drone.x < 50 or drone.x > MAP_SIZE_0 - 50 or drone.y < 50 or drone.y > MAP_SIZE_1 - 50:
                breakdown['reward_border'] = -0.2
                rewards[idx] += breakdown['reward_border']

            breakdown['total'] = float(rewards[idx])
            self.reward_breakdown_log.append(breakdown)

            if self.debug:
                print(f"[Drone {idx}] Reward breakdown:", breakdown)

            self.last_positions[drone.id] = (drone.x, drone.y)

        return rewards

    def get_reward_log(self):
        """返回奖励日志（用于绘图或后期分析）"""
        return self.reward_breakdown_log
    


class CurriculumReward:
    def __init__(self, drones: list, actions: list, task_type="task1", debug=False):
        self.drones = drones
        self.actions = actions
        self.task_type = task_type
        self.debug = debug
        self.reward_breakdown_log = []
        self.last_positions = {drone.id: (drone.x, drone.y) for drone in drones}

    def _get_enemies(self, drone):
        return [d for d in self.drones if d.teamcode != drone.teamcode and d.alive and (d.x-drone.x)**2+(d.y-drone.y)**2 <= 200**2]

    def update_and_return(self):
        rewards = np.zeros(len(self.drones))

        for idx, drone in enumerate(self.drones):
            breakdown = {
                'id': drone.id,
                'team': drone.team,
                'alive': int(drone.alive),
                'reward_survive': 0.0,
                'reward_kill': 0.0,
                'reward_find_enemy': 0.0,
                'reward_facing': 0.0,
                'reward_fire': 0.0,
                'reward_border': 0.0,
                'reward_distance': 0.0,
                'reward_team_cohesion': 0.0,
                'total': 0.0
            }

            if not drone.alive:
                rewards[idx] -= 1.0
                breakdown['total'] = -1.0
                self.reward_breakdown_log.append(breakdown)
                continue

            # ===== 通用奖励项 =====
            rewards[idx] += 0.1
            breakdown['reward_survive'] = 0.1

            action = self.actions[idx]
            enemies = self._get_enemies(drone)

            # ===== 针对阶段的奖励逻辑 =====
            if self.task_type == "task1":
                # 击杀奖励（任何敌人）
                for enemy in enemies:
                    if not enemy.alive:
                        rewards[idx] += 1.0
                        breakdown['reward_kill'] += 1.0

                # 发现敌人奖励
                if len(enemies)>0:
                    rewards[idx] += 0.1
                    breakdown['reward_find_enemy'] += 0.1

                # 面朝敌人奖励
                if enemies:
                    nearest = min(enemies, key=lambda e: (e.x - drone.x) ** 2 + (e.y - drone.y) ** 2)
                    dx, dy = nearest.x - drone.x, nearest.y - drone.y
                    target_angle = math.atan2(dy, dx)
                    angle_diff = abs((drone.orientation - target_angle + math.pi) % (2 * math.pi) - math.pi)
                    facing_reward = 0.2 * (1 - angle_diff / math.pi)
                    rewards[idx] += facing_reward
                    breakdown['reward_facing'] = facing_reward

            elif self.task_type == "task2":
                # 击杀长机奖励（假设 id == 0 为长机）
                for enemy in enemies:
                    if not enemy.alive and enemy.id == 0:
                        rewards[idx] += 3.0
                        breakdown['reward_kill'] += 3.0

                # 队形协作奖励（鼓励接近队友）
                allies = [d for d in self.drones if d.teamcode == drone.teamcode and d.id != drone.id and d.alive]
                if allies:
                    avg_dist = sum(math.hypot(a.x - drone.x, a.y - drone.y) for a in allies) / len(allies)
                    cohesion = 0.05 * (1 - min(avg_dist / 500, 1.0))
                    rewards[idx] += cohesion
                    breakdown['reward_team_cohesion'] = cohesion

            elif self.task_type == "task3":
                # 控制靠近敌人奖励（鼓励主动接敌）
                if enemies:
                    nearest = min(enemies, key=lambda e: (e.x - drone.x) ** 2 + (e.y - drone.y) ** 2)
                    dist = math.hypot(nearest.x - drone.x, nearest.y - drone.y)
                    dist_reward = 0.1 * (1 - min(dist / 300, 1.0))
                    rewards[idx] += dist_reward
                    breakdown['reward_distance'] = dist_reward

            # ===== 通用发射奖励 =====
            if drone.fire_cooltime <= 0 and action[2] > 0:
                rewards[idx] += 0.5
                breakdown['reward_fire'] = 0.5

            # ===== 通用边界惩罚 =====
            if drone.x < 50 or drone.x > MAP_SIZE_0 - 50 or drone.y < 50 or drone.y > MAP_SIZE_1 - 50:
                rewards[idx] -= 0.2
                breakdown['reward_border'] = -0.2

            breakdown['total'] = rewards[idx]
            self.reward_breakdown_log.append(breakdown)

            if self.debug:
                print(f"[Drone {idx}] Reward breakdown:", breakdown)

            self.last_positions[drone.id] = (drone.x, drone.y)

        return rewards

    def get_reward_log(self):
        return self.reward_breakdown_log



class DroneRewardSector:
    def __init__(self, drones, actions, debug=False):
        self.drones = drones
        self.actions = actions
        self.debug = debug
        self.reward_log = []

    def _get_enemies(self, drone):
        return [d for d in self.drones if d.teamcode != drone.teamcode and d.alive]

    def _get_enemies_in_sight(self, drone):
        return [d for d in self.drones if d.teamcode != drone.teamcode and d.alive and (d.x-drone.x)**2+(d.y-drone.y)**2 <= RADIUS**2]

    def _in_sector(self, enemy, drone):
        # 判断是否在扇形区域（你应已有此函数）
        self_posi = (drone.x, drone.y)
        enemy_posi = (enemy.x, enemy.y)
        return is_in_sector(enemy_posi, self_posi, drone.orientation)

    def update_and_return(self):
        rewards = np.zeros(len(self.drones))

        for idx, drone in enumerate(self.drones):
            reward = 0.0
            if not drone.alive:
                rewards[idx] = -1.0
                self.reward_log.append({'id': drone.id, 'total': -1.0})
                continue

            # reward += 0.1  # 存活奖励

            enemies = self._get_enemies_in_sight(drone)
            if enemies:
                nearest = min(enemies, key=lambda e: (e.x - drone.x) ** 2 + (e.y - drone.y) ** 2)
                dist = math.hypot(nearest.x - drone.x, nearest.y - drone.y)
                target_angle = math.atan2(nearest.y - drone.y, nearest.x - drone.x)
                angle_diff = abs((drone.orientation - target_angle) % (2 * math.pi))

                # 距离 shaping：靠近敌人（非盲目冲刺）
                if 100 < dist < 400:
                    reward += 0.1 * (1 - abs(dist - 250) / 150)

                # 面向敌人奖励
                reward += 0.1 * (1 - angle_diff / math.pi)

                # 惩罚：暴露在敌人扇形攻击范围
                for enemy in enemies:
                    if self._in_sector(drone, enemy):
                        reward += 1.0
                for enemy in enemies:
                    if self._in_sector(enemy, drone):
                        reward -= 0.5
                        break

            # 边界惩罚
            if drone.x < 50 or drone.x > MAP_SIZE_0 - 50 or drone.y < 50 or drone.y > MAP_SIZE_1 - 50:
                reward -= 0.3

            rewards[idx] = reward
            self.reward_log.append({'id': drone.id, 'total': reward})

            if self.debug:
                print(f"[Drone {drone.id}] Reward: {reward:.3f}")

        return rewards

    def get_reward_log(self):
        return self.reward_log


