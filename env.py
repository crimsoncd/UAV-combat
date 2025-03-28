import pygame
import numpy as np
from pygame.locals import *
import sys
import math

from copy import deepcopy
import json
from env_utils import DroneReward, rewind, nearest_direction

class Drone:
    def __init__(self, drone_id, team, x, y):
        self.id = drone_id
        self.team = team  # 'red' 或 'blue'
        self.teamcode = 0 if team=='red' else 1
        self.x = x
        self.y = y
        self.alive = True
        self.fire_cooltime = 0
        self.max_speed = 5
        self.vx = 0
        self.vy = 0
        self.orientation = 0

    def _update(self, sa, ca, vs):

        vs = 0 if vs<0 else vs
        self.vx = self.max_speed * vs * sa
        self.vy = self.max_speed * vs * ca

        self.x += self.vx
        self.y += self.vy
        self.orientation = self.vy / self.vx if self.vx!=0 else 999

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
            self.x += self.speed * self.vx
            self.y += self.speed * self.vy
            self.life -= 1
            if self.life <= 0:
                self._reset()
        # print("Missile index", self.id, " state:", self.exist)

    def _shootout(self, start_x, start_y, vx, vy):
        self.exist = True
        self.x = start_x
        self.y = start_y
        self.vx = vx
        self.vy = vy
        self.orientation = self.vy / self.vx if self.vx!=0 else 999
        self.life = self.lifelong

    def _collide(self):
        self._reset()

class BattleEnv:
    def __init__(self, red_agents=5, blue_agents=5, auto_record=True):
        # 初始化参数
        self.red_agents = red_agents
        self.blue_agents = blue_agents
        self.total_agents = red_agents + blue_agents
        self.map_size = (750, 750)
        self.drone_speed = 3
        self.max_speed = 5
        self.missile_speed = 8
        self.fire_cooldown = 30
        self.missile_ttl = 30
        self.attack_radius = 15
        
        # 初始化无人机
        self.drones = []
        self._init_drones()
        
        # 导弹列表
        self.missiles = []
        self._init_missles()
        
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
                drone.x = 50
                drone.y = np.random.uniform(0, self.map_size[1])
            else:
                drone.x = self.map_size[0] - 50
                drone.y = np.random.uniform(0, self.map_size[1])
            drone.alive = True
            drone.fire_cooltime = 0
        for missile in self.missiles:
            missile._reset()
        self.battle_idx = 0
        return self._get_state()

    def _get_state(self):
        """获取全局状态（简化版）"""
        state = []
        for drone in self.drones:
            # state += [drone.x, drone.y, drone.alive, drone.fire_cooltime]
            state += [drone.x, drone.y, drone.alive, drone.fire_cooltime]
        for missile in self.missiles:
            state += [missile.x, missile.y, missile.orientation, int(missile.exist)]
        return np.array(state, dtype=np.float32)
    
    def _get_obs(self, idx):
        obs = []
        for drone in self.drones:
            if drone.teamcode == self.drones[idx].teamcode:
                obs += [drone.x, drone.y, drone.alive, drone.fire_cooltime]
            else:
                if (drone.x-self.drones[idx].x)**2 + (drone.y-self.drones[idx].y)**2 <= 200**2:
                    obs += [drone.x, drone.y, drone.alive, drone.fire_cooltime]
                else:
                    obs += [0, 0, 0, 0]
        for missile in self.missiles:
            obs += [missile.x, missile.y, missile.orientation, int(missile.exist)]
        return np.array(obs, dtype=np.float32)
    
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
        epoch_record = {
            "epoch": epoch_num,
            "frames": self.records
        }
        with open(record_path, 'a') as f:
            f.write(json.dumps(epoch_record) + '\n')  # 注意换行符
        self.records = []

    def set_frame_data(self, frame_data):
        for obj in frame_data:
            if obj["type"]=="Drone":
                self.drones[obj["id"]].__dict__.update(deepcopy(obj["state"]))
            elif obj["type"]=="Missile":
                self.missiles[obj["id"]].__dict__.update(deepcopy(obj["state"]))
    
    def induce_step(self, idx):
        self_drone = self.drones[idx]
        target_drone = None
        lst_distance = 1000000

        for enemy in self.drones:
            if enemy.teamcode!=self_drone.teamcode and enemy.x**2+enemy.y**2>0:
                distance = (self_drone.x - enemy.x)**2 + (self_drone.y - enemy.y)**2
                if distance <= lst_distance:
                    lst_distance = distance
                    target_drone = enemy

        if target_drone==None:
            return 1 if self_drone.teamcode==0 else 8

        facing = nearest_direction(self_drone.x, self_drone.y, target_drone.x, target_drone.y)
        if distance <= 40000:
            if self_drone.orientation!=facing:
                return facing + 1
            else:
                if self_drone.fire_cooltime<=0:
                    return 9
                else:
                    return 0
        else:
            return facing + 1

    def step(self, actions):
        """执行动作"""
        rewards = np.zeros(self.total_agents)
        
        # 处理无人机移动和发射
        for idx, drone in enumerate(self.drones):
            if not drone.alive:
                continue

            # [sin alpha, cos alpha, velosity scale, shoot]
            sa, ca, vs, sh = actions[idx]
            action_shoot = True if sh>0 else False

            drone._update(sa, ca, vs)
            
            # 出地图边界
            if drone.x<0 or drone.x>self.map_size[0] or drone.y<0 or drone.y>self.map_size[1]:
                drone.alive = False

            # 检查碰撞
            for other_idx, other_drone in enumerate(self.drones):
                if idx!=other_idx and abs(drone.x-other_drone.x)<3 and abs(drone.y-other_drone.y)<3:
                    drone.alive = False
                    other_drone.alive = False
            
            # 处理发射
            if action_shoot and drone.fire_cooltime <= 0:
                self.missiles[idx]._shootout(drone.x, drone.y, drone.vx, drone.vy)
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

                    missile._collide()
                    
                    # 更新奖励
                    rewards[missile.id] += 50  # 攻击者奖励
                    rewards[drone.id] -= 50           # 被攻击者惩罚
                    drone.alive = False
                    break
                else:
                    rewards[missile.id] += 1

        # 保存记录
        if self.auto_record:
            self._record_frame()
            self.frame_idx += 1

        DronesReward = DroneReward(self.drones, actions)
        drones_rewards = DronesReward.update_and_return()
        rewards = np.add(rewards, drones_rewards)

        # 检查终止条件
        done = not (any(d.alive for d in self.drones[:self.red_agents]) and 
                    any(d.alive for d in self.drones[self.red_agents:]))
        
        # rewards = rewards[:num_drones//2]
        rewards = rewards[:]
        
        return self._get_state(), rewards, done, {}
    


    def render(self):
        """可视化"""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.map_size)
            self.clock = pygame.time.Clock()
            self.drone_img = pygame.image.load(r'assets\drone.png').convert_alpha()
            self.drone_img = pygame.transform.scale(self.drone_img, (40, 40))
            
        # self.screen.fill((255, 255, 255))
        self.screen.fill(self.color_darkbg)
        
        # 绘制无人机
        for drone in self.drones:
            if not drone.alive:
                continue
            color = self.color_red_team if drone.team == 'red' else self.color_blue_team
            # pygame.draw.circle(self.screen, color, (int(drone.x), int(drone.y)), 10)
            # print(drone.orientation)
            rotation = np.arctan2(drone.vy, drone.vx)
            rotated_img = pygame.transform.rotate(self.drone_img, rotation*360 - 90)  # 转换为顺时针旋转
            rect = rotated_img.get_rect(center=(drone.x, drone.y))
            self.screen.blit(rotated_img, rect)
            # pygame.draw.circle(self.screen, color, (drone.x, drone.y), rect.width//2 + 5, 2 ) # 线宽)
            pygame.draw.circle(self.screen, color, (int(drone.x), int(drone.y)), 200, width=1)
            
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
    env = BattleEnv(red_agents=1, blue_agents=1, auto_record=False)
    state = env.reset()

    step = 0
    
    while True and step<500:
        # 生成随机动作（连续动作空间）
        actions = []
        actions.append([1, 1, 1, 1])
        for unit in range(1):
            drone_action = []
            for i in range(4):
                a = np.random.random() * 2 - 1
                drone_action.append(a)
            actions.append(drone_action)
        
        # print("actions:", actions)
        
        # 执行动作
        next_state, rewards, done, _ = env.step(actions)
        step += 1
        
        # 渲染
        env.render()
        
        # 检查终止条件
        if done:
            state = env.reset()
            
        # 控制台输出
        # print(f"当前奖励: {rewards}")