import pygame
import numpy as np
from pygame.locals import *
import sys
import math

from copy import deepcopy
import json
import env_utils

import os
import csv

from config import *
ACTION_DIM = 3


def relative_polar(x0, y0, x1, y1, orientation=math.pi / 2, standard=False):

    dx = x1 - x0
    dy = y1 - y0
    
    r = math.sqrt(dx**2 + dy**2)  # 半径
    theta_raw = math.atan2(dy, dx)  # 原始角度（相对于 x 轴）
    theta = theta_raw - orientation  # 调整后的角度（以 orientation 为基准）
    
    # 确保角度在 [-π, π] 范围内
    if theta < -math.pi:
        theta += 2 * math.pi
    elif theta > math.pi:
        theta -= 2 * math.pi

    if standard:
        r = r / (MAP_SIZE_0**2 + MAP_SIZE_1**2)
        theta = theta / math.pi
    
    return [r, theta]



class Drone:
    def __init__(self, drone_id, team, x, y):
        # 固定量
        self.id = drone_id
        self.team = team  # 'red' 或 'blue'
        self.teamcode = 0 if team=='red' else 1

        # 环境量
        self.fire_cooltime = 0
        self.alive = True

        # 状态量: x, y, alpha, v, w
        self.x = x
        self.y = y
        self.orientation = float(0) if self.teamcode==0 else float(np.pi)
        self.v = 0
        self.w = 0

    def _update(self, a, phi):

        # a, phi: accelerate, angle accelerate
        self.v += np.clip(a, -MAX_ACCELERATE, MAX_ACCELERATE)
        self.v = np.clip(self.v, 0, MAX_SPEED)
        self.w += np.clip(phi, -MAX_ANGLE_ACCE, MAX_ANGLE_ACCE)
        self.w = np.clip(self.w, -MAX_ANGLE_SPEED, MAX_ANGLE_SPEED)

        # Formalize
        self.v = float(self.v)
        self.w = float(self.w)

        # alpha' = alpha + w * t
        self.orientation += self.w
        self.orientation = float(self.orientation % (2*np.pi))

        # Change in position
        self.x += self.v * np.cos(self.orientation)
        self.y += self.v * np.sin(self.orientation)



class Missile:
    def __init__(self, idx, teamcode):
        self.id = idx
        self.exist = False
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.orientation = 0
        self.teamcode = teamcode
        self.speed = 12
        self.lifelong = 20
        self.life = 0

    def _reset(self):
        self.exist = False
        self.x = 0
        self.y = 0
        self.orientation = 0
        self.life = 0
        self.vx = 0
        self.vy = 0

    def _update(self):
        if self.exist:
            self.x += self.vx
            self.y += self.vy
            self.life -= 1
            if self.life <= 0:
                self._reset()
        # print("Missile index", self.id, " state:", self.exist)

    def _shootout(self, start_x, start_y, orientation):
        self.exist = True
        self.x = start_x
        self.y = start_y
        self.vx = self.speed * np.cos(orientation)
        self.vy = self.speed * np.sin(orientation)
        self.vx, self.vy = float(self.vx), float(self.vy)
        self.orientation = orientation
        self.life = self.lifelong

    def _collide(self):
        self._reset()

class BattleEnv:
    def __init__(self, red_agents=5, blue_agents=5, auto_record=True, developer_tools=False,
                 margin_crash=True, collision_crash=True):
        # 初始化参数
        self.red_agents = red_agents
        self.blue_agents = blue_agents
        self.total_agents = red_agents + blue_agents
        self.map_size = (MAP_SIZE_0, MAP_SIZE_1)
        self.drone_speed = 3
        self.max_speed = 5
        self.missile_speed = 8
        self.fire_cooldown = 30
        self.missile_ttl = 30
        self.attack_radius = 15
        self.margin_crash = margin_crash
        self.collision_crash = collision_crash
        
        # 初始化无人机
        self.drones = []
        self._init_drones()
        
        # 导弹列表
        self.missiles = []
        self._init_missles()

        self.victory_point = [0.0, 0.0]
        
        # 渲染相关
        self.screen = None
        self.clock = None
        # self.drone_img = pygame.image.load(r'train\F_commu\assets\drone.png').convert_alpha()
        # self.drone_img = pygame.transform.scale(self.drone_img, (40, 40))
        self.color_darkbg = (30, 30, 30)
        self.color_red_team = (255, 50, 50)
        self.color_blue_team = (50, 50, 255)

        # Record
        self.auto_record = auto_record
        if auto_record:
            self.battle_idx = 0
            self.frame_idx = 0
            self.records = []
            self.reward_records = []

        # Developer Tools
        self.develop = False
        if developer_tools:
            self.develop = True
            self.trail_red = []
            self.trail_blue = []

    def _init_drones(self):
        """初始化无人机位置"""
        for i in range(self.red_agents):
            x = 100
            y = (self.map_size[1]/self.red_agents)*(i)
            # print("Initialized red:", x, y)
            self.drones.append(Drone(i, 'red', x, y))
        for i in range(self.blue_agents):
            x = self.map_size[0] - 100
            y = (self.map_size[1]/self.blue_agents)*(i)
            # print("Initialized blue:", x, y)
            self.drones.append(Drone(self.red_agents+i, 'blue', x, y))

    def _init_missles(self):
        for i in range(self.red_agents):
            self.missiles.append(Missile(i, teamcode=0))
        for i in range(self.blue_agents):
            self.missiles.append(Missile(i+self.red_agents, teamcode=1))

    def reset(self):
        """重置环境"""
        for drone in self.drones:
            if drone.team == 'red':
                drone.x = np.random.uniform(50, self.map_size[0] - 50)
                drone.y = np.random.uniform(50, self.map_size[1] - 50)
            else:
                drone.x = np.random.uniform(50, self.map_size[0] - 50)
                drone.y = np.random.uniform(50, self.map_size[1] - 50)
            drone.alive = True
            drone.fire_cooltime = 0
        for missile in self.missiles:
            missile._reset()
        self.battle_idx = 0
        self.victory_point = [0.0, 0.0]
        if self.develop:
            self.trail_blue = []
            self.trail_red = []
        return self._get_state()

    def _get_state(self):
        """获取全局状态（简化版）"""
        state = []
        for drone in self.drones:
            # state += [drone.x, drone.y, drone.alive, drone.fire_cooltime]
            state += [drone.x, drone.y, drone.alive, drone.v, drone.w, drone.orientation]
        for missile in self.missiles:
            state += [missile.x, missile.y, missile.orientation, int(missile.exist)]
        return np.array(state, dtype=np.float32)
    
    def _search_enemies(self, max_enemy_obs=3):
        search = []
        for idx in range(self.red_agents):
            local_search = []
            visible_enemies = self._get_enemy_in_ally_sight(idx)
            for i in range(max_enemy_obs):
                if i < len(visible_enemies):
                    e = visible_enemies[i]
                    local_search.append((e.x, e.y))
                else:
                    local_search.append(None)
            search.append(local_search)
        return search

    
    def _get_obs(self, idx, max_enemy_obs=3, full=False, polar=False):
        obs = []
        drone = self.drones[idx]
        # Self
        obs += [drone.x / MAP_SIZE_0, drone.y / MAP_SIZE_1,
                # drone.alive,
                drone.v / MAX_SPEED, drone.w / MAX_ANGLE_SPEED,
                drone.orientation / (np.pi)]
        # Enemy
        if full:
            enemies = [d for d in self.drones if d.teamcode != drone.teamcode]
            for e in enemies:
                if e.alive:
                    if polar:
                        coord1, coord2 = relative_polar(drone.x, drone.y, e.x, e.y, standard=True)
                    else:
                        coord1, coord2 = e.x / MAP_SIZE_0, e.y / MAP_SIZE_1
                    obs += [1.0, coord1, coord2, e.v / MAX_SPEED, e.orientation / (np.pi) ]
                else:
                    obs += [0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            visible_enemies = self._get_enemy_in_sight(idx)
            for i in range(max_enemy_obs):
                if i < len(visible_enemies):
                    e = visible_enemies[i]
                    if polar:
                        coord1, coord2 = relative_polar(drone.x, drone.y, e.x, e.y, standard=True)
                    else:
                        coord1, coord2 = e.x / MAP_SIZE_0, e.y / MAP_SIZE_1
                    obs += [1.0, coord1, coord2, e.v / MAX_SPEED, e.orientation / (np.pi) ]
                else:
                    obs += [0.0, 0.0, 0.0, 0.0, 0.0]  # 不可观测敌人
        return obs
    
    def _get_obs_all(self, max_enemy_obs=3, full=False, polar=False):
        obs = []
        for i in range(self.total_agents):
            obs.append(self._get_obs(i, max_enemy_obs=max_enemy_obs, full=full, polar=polar))
        return obs
    
    def _get_obs_side(self, side=0, max_enemy_obs=3, full=False, polar=False):
        obs = []
        if side==0:
            side_range = [_ for _ in range(self.red_agents)]
        else:
            side_range = [_+self.red_agents for _ in range(self.blue_agents)]
        for i in side_range:
            obs.append(self._get_obs(i, max_enemy_obs=max_enemy_obs, full=full, polar=polar))
        return obs
    
    def _get_obs_self(self, idx):
        obs = []
        drone = self.drones[idx]
        # Self
        obs += [drone.x / MAP_SIZE_0, drone.y / MAP_SIZE_1,
                # drone.alive,
                drone.v / MAX_SPEED, drone.w / MAX_ANGLE_SPEED,
                drone.orientation / (np.pi)]
        return obs

    
    def _update_victory_point(self):
        for drone in self.drones:
            if drone.alive:
                self.victory_point[drone.teamcode] += 1.0
            enemies = self._get_enemy_in_sight(drone.id)
            if enemies:
                nearest = min(enemies, key=lambda e: (e.x - drone.x) ** 2 + (e.y - drone.y) ** 2)
                # dist = math.hypot(nearest.x - drone.x, nearest.y - drone.y)
                target_angle = math.atan2(nearest.y - drone.y, nearest.x - drone.x)
                angle_diff = abs((drone.orientation - target_angle) % (2 * math.pi))
                self.victory_point[drone.teamcode] += 0.1 * (1 - angle_diff / math.pi)
    
    def _get_all_drones(self):
        return self.drones
    
    def _get_random_drone(self, team=0):
        target_drone = np.random.choice([u for u in self.drones if u.teamcode==team])
        return target_drone
    
    def _get_allies(self, idx):
        drone = self.drones[idx]
        return [d for d in self.drones if d.teamcode == drone.teamcode and d.alive]
    
    def _get_enemies(self, idx):
        drone = self.drones[idx]
        return [d for d in self.drones if d.teamcode != drone.teamcode and d.alive]
    
    def _get_enemy_in_sight(self, idx):
        drone = self.drones[idx]
        return [d for d in self.drones if d.teamcode != drone.teamcode and d.alive and (d.x-drone.x)**2+(d.y-drone.y)**2 <= RADIUS**2]
    
    def _get_enemy_in_ally_sight(self, idx):
        allies = self._get_allies(idx)
        enemies = self._get_enemies(idx)

        enemies_in_sight = []
        for e in enemies:
            if e.alive:
                for a in allies:
                    if (e.x - a.x)**2 + (e.y - a.y)**2 <= RADIUS **2:
                        enemies_in_sight.append(e)
                        break
        return enemies_in_sight
    
    def _get_enemies_position(self, side=0):
        positions = [(d.x, d.y) for d in self.drones if d.teamcode!=side]
        return positions
    
    def _get_random_drone_position(self, team=0):
        target_drone = np.random.choice([u for u in self.drones if u.teamcode==team])
        return (target_drone.x, target_drone.y)
    
    def _record_frame(self):
        frame = {
            "battle_index": self.battle_idx,
            "frame_index": self.frame_idx,
            "objects": []
        }

        for obj in self.drones:
            frame["objects"].append({
                "type": "Drone",
                "id": obj.id,
                "state": deepcopy(obj.__dict__)
            })

        for obj in self.missiles:
            frame["objects"].append({
                "type": "Missile",
                "id": obj.id,
                "state": deepcopy(obj.__dict__)
            })

        self.records.append(frame)

    def save_and_clear(self, epoch_num, record_path):
        if not self.auto_record:
            return
        epoch_record = {
            "epoch": epoch_num,
            "frames": self.records
        }
        with open(record_path, 'a') as f:
            f.write(json.dumps(epoch_record) + '\n')  # 注意换行符
        self.records = []

    def _record_reward(self, reward_breakdown_list):
        for breakdown in reward_breakdown_list:
            frame = {
                "battle_index": self.battle_idx,
                "frame_index": self.frame_idx
            }
            frame.update(breakdown)
            self.reward_records.append(frame)

    def save_and_clear_rewards(self, epoch_num, record_path):
        if not self.auto_record:
            return
        file_exists = os.path.isfile(record_path)

        for frame_record in self.reward_records:
            line_data = {"epoch": epoch_num} | frame_record
            with open(record_path, mode='a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=line_data.keys())   
                if not file_exists:
                    writer.writeheader()  # 写入表头
                    file_exists = True   
                writer.writerow(line_data)  # 写入数据行

        self.reward_records = []

    def set_frame_data(self, frame_data):
        for obj in frame_data:
            if obj["type"]=="Drone":
                self.drones[obj["id"]].__dict__.update(deepcopy(obj["state"]))
                if self.develop:
                    if obj["state"]["teamcode"]==0:
                        self.trail_red.append([int(obj["state"]["x"]), int(obj["state"]["y"])])
                    elif obj["state"]["teamcode"]==1:
                        self.trail_blue.append([int(obj["state"]["x"]), int(obj["state"]["y"])])
            elif obj["type"]=="Missile":
                self.missiles[obj["id"]].__dict__.update(deepcopy(obj["state"]))
    
    def induce_step(self, idx):

        drone = self.drones[idx]
        return env_utils.control_strategy_Expert(drone, self.drones)
    
    def decide_outcome(self):
        red_alive = len([d for d in self.drones if d.teamcode==0 and d.alive==True])
        blue_alive = len([d for d in self.drones if d.teamcode==1 and d.alive==True])
        if red_alive == blue_alive:
            return 'draw'
        return 'red win' if red_alive > blue_alive else 'blue win'
        # return 'blue win' if blue_alive > red_alive else 'red win'
        # if self.victory_point[0] > self.victory_point[1]:
        #     return 'red win'
        # elif self.victory_point[0] < self.victory_point[1]:
        #     return 'blue win'
        # else:
        #     return 'draw'
        


    def step(self, actions, reward_type=None, half_reward=False):
        """执行动作"""
        #  rewards = np.zeros(self.total_agents)
        shoot_rewards = np.zeros(self.total_agents)
        
        # 处理无人机移动和发射
        for idx, drone in enumerate(self.drones):
            if not drone.alive:
                continue

            # [sin alpha, cos alpha, velosity scale, shoot]
            a, phi = actions[idx][0], actions[idx][1]
            sh = actions[idx][2]
            action_shoot = True if sh>0 else False

            drone._update(a * MAX_ACCELERATE, phi * MAX_ANGLE_ACCE)

            # 记录轨迹
            if self.develop:
                if drone.teamcode==0:
                    self.trail_red.append([int(drone.x), int(drone.y)])
                else:
                    self.trail_blue.append([int(drone.x), int(drone.y)])
            
            # 出地图边界
            if self.margin_crash:
                if drone.x<0 or drone.x>self.map_size[0] or drone.y<0 or drone.y>self.map_size[1]:
                    drone.alive = False

            # 检查碰撞
            if self.collision_crash:
                for other_idx, other_drone in enumerate(self.drones):
                    if idx!=other_idx and abs(drone.x-other_drone.x)<10 and abs(drone.y-other_drone.y)<10:
                        drone.alive = False
                        other_drone.alive = False

            # 检查攻击范围
            # enemies = self._get_enemies(idx)
            # for enemy in enemies:
            #     drone_posi = (drone.x, drone.y)
            #     enemy_posi = (enemy.x, enemy.y)
            #     if is_in_sector(drone_posi, enemy_posi, drone.orientation, ATTACK_ALPHA, ATTACK_R) and enemy.teamcode==1:
            #         enemy.alive = False
            #         shoot_rewards[drone.id] += 10
            #         shoot_rewards[enemy.id] -= 10
            #     if is_in_sector(enemy_posi, drone_posi, enemy.orientation, ATTACK_ALPHA, ATTACK_R) and enemy.teamcode==1:
            #         drone.alive = False
            #         shoot_rewards[drone.id] -= 10
            #         shoot_rewards[enemy.id] += 10

            # 靠近敌人
            # enemies_in_sight = self._get_enemy_in_sight(idx)
            # if enemies_in_sight:
            #     nearest = min(enemies_in_sight, key=lambda e: (e.x - drone.x) ** 2 + (e.y - drone.y) ** 2)
            #     dist = math.hypot(nearest.x - drone.x, nearest.y - drone.y)
            #     dist_reward = 0.1 * (1 - min(dist / 300, 1.0))
            #     shoot_rewards[drone.id] += dist_reward
            
            # 处理发射
            if action_shoot and drone.fire_cooltime <= 0:
                self.missiles[idx]._shootout(drone.x, drone.y, drone.orientation)
                drone.fire_cooltime = self.fire_cooldown
            
            if drone.fire_cooltime > 0:
                drone.fire_cooltime -= 1

        # 处理导弹
        for missile in self.missiles:
            # 更新位置
            missile._update()
                
            # 碰撞检测
            for drone in self.drones:
                if (drone.teamcode != missile.teamcode and 
                    drone.alive and 
                    np.hypot(drone.x-missile.x, drone.y-missile.y) < self.attack_radius):
                    shoot_rewards[missile.id] += 10
                    shoot_rewards[drone.id] -= 10
                    missile._collide()
                    drone.alive = False

        self._update_victory_point()
        
        TypeReward = None
        if reward_type==None or reward_type=="half":
            TypeReward = env_utils.CurriculumReward(self.drones, actions, "task1")
            rewards = TypeReward.update_and_return()
        elif 'task' in reward_type:
            TypeReward = env_utils.CurriculumReward(self.drones, actions, reward_type)
            rewards = TypeReward.update_and_return()
        elif 'sector' in reward_type:
            TypeReward = env_utils.DroneRewardSector(self.drones, actions)
            rewards = TypeReward.update_and_return()
        elif 'only_shoot' in reward_type:
            rewards = shoot_rewards

        # rewards = TypeReward.update_and_return()

        # 保存记录
        if self.auto_record:
            self._record_frame()
            if TypeReward:
                self._record_reward(TypeReward.get_reward_log())
            self.frame_idx += 1

        # 检查终止条件
        done = not (any(d.alive for d in self.drones[:self.red_agents]) and 
                    any(d.alive for d in self.drones[self.red_agents:]))
        
        obs_n = self._get_obs_all()
        
        if half_reward:
            rewards = rewards[:self.total_agents//2]
            # drones_rewards = drones_rewards[:self.total_agents//2]
            # obs_n = obs_n[:self.total_agents//2]
        
        return obs_n, rewards, done, {}
    


    def render(self, show_trail=False):
        """可视化"""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.map_size)
            self.clock = pygame.time.Clock()
            self.drone_img = pygame.image.load(r'assets\drone.png').convert_alpha()
            self.drone_img = pygame.transform.scale(self.drone_img, (30, 30))
            self.drone_img_a = pygame.image.load(r'assets\drone_purple.png').convert_alpha()
            self.drone_img_a = pygame.transform.scale(self.drone_img_a, (30, 30))
            
        self.screen.fill((255, 255, 255))
        # self.screen.fill(self.color_darkbg)

        for drone in self.drones:
            if not drone.alive:
                continue
            # 绘制无人机扇形
            fan_color = (255, 182, 193) if drone.teamcode==0 else (173, 216, 230)
            draw_fan_sector(self.screen, (drone.x, drone.y), drone.orientation, ATTACK_ALPHA, ATTACK_R, fan_color)
        
        # 绘制无人机
        for drone in self.drones:
            # if not drone.alive:
            #     continue
            color = self.color_red_team if drone.team == 'red' else self.color_blue_team
            # pygame.draw.circle(self.screen, color, (int(drone.x), int(drone.y)), 10)
            # print(drone.orientation)
            # rotation = np.arctan2(-drone.vy, drone.vx)
            rotation = drone.orientation
            to_draw = self.drone_img if drone.team=="red" else self.drone_img_a
            # rotated_img = pygame.transform.rotate(to_draw, rotation*180/np.pi + 90)  # 转换为顺时针旋转
            rotated_img = pygame.transform.rotate(to_draw, -rotation*180/np.pi -90)
            rect = rotated_img.get_rect(center=(drone.x, drone.y))
            self.screen.blit(rotated_img, rect)
            # pygame.draw.circle(self.screen, color, (drone.x, drone.y), rect.width//2 + 5, 2 ) # 线宽)
            pygame.draw.circle(self.screen, color, (int(drone.x), int(drone.y)), 200, width=1)
            # 绘制无人机扇形
            # fan_color = (150, 0, 0) if drone.teamcode==0 else (0, 0, 150)
            # draw_fan_sector(self.screen, (drone.x, drone.y), drone.orientation, ATTACK_ALPHA, ATTACK_R, fan_color)


        # 绘制轨迹
        if self.develop and show_trail:
            for footstep in self.trail_red:
                pygame.draw.circle(self.screen, (255, 150, 0), footstep, 1)
            for footstep in self.trail_blue:
                pygame.draw.circle(self.screen, (0, 255, 255), footstep, 1)
            
        # 绘制导弹
        for missile in self.missiles:
            if missile.exist:
                color = (255,150,0) if missile.teamcode == 0 else (0,255,255)
                pygame.draw.circle(self.screen, color, (int(missile.x), int(missile.y)), 5)
            
        pygame.display.flip()
        self.clock.tick(30)
        
        # 处理退出事件
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

# 测试代码
if __name__ == "__main__":

    import time

    num_agents = 3

    env = BattleEnv(red_agents=num_agents, blue_agents=num_agents, auto_record=False, developer_tools=True)
    state = env.reset()

    step = 0
    
    while True and step<1000:
        # 生成随机动作（连续动作空间）
        actions = []
        


        for i in range(num_agents):
        # 一个飞机使用策略
            self_drone = env.drones[i]
            all_drones = env._get_all_drones()
            # self_drone = env._get_random_drone(1)
            # target_x, target_y = env._get_random_drone_position(0)
            # actions.append(env_utils.control_strategy(self_drone, target_x, target_y))
            # actions.append(env_utils.control_to_attack(self_drone, target_x, target_y))
            # actions.append(env_utils.control_strategy_C(self_drone, target_x, target_y))
            actions.append(env_utils.control_strategy_Expert(self_drone, all_drones))

        for i in range(num_agents):
            # 一个飞机随机飞行
            actions.append(np.random.random(3) * 2 - 1)
            # actions.append([0, 0, 0])


        # print("actions:", actions)
        
        # 执行动作
        next_state, rewards, done, _ = env.step(actions)
        step += 1
        
        # 渲染
        env.render(show_trail=True)
        time.sleep(0.05)
        
        # 检查终止条件
        if done:
            state = env.reset()
            
        # 控制台输出
        # print(f"当前奖励: {rewards}")