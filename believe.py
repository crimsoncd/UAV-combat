import numpy as np
from scipy.optimize import linear_sum_assignment

import time

class BlackBox:
    def __init__(self, num_enemies=5):
        self.num_enemies = num_enemies
        self.timestamp = 0

        self.dead_waiting = 0
        
        # 初始化存储敌人状态的空间
        self.memory = {
            idx: {"x": None, "y": None, "v": None, "orientation": None, "alive": False, "last_seen": -1}
            for idx in range(num_enemies)
        }

    def reset(self):
        self.memory = {
            idx: {"x": None, "y": None, "v": None, "orientation": None, "alive": False, "last_seen": -1}
            for idx in range(self.num_enemies)
        }

    def _hungarian_match(self, stored_positions, new_observations):
        """ 使用匈牙利算法对当前观测和存储信息进行匹配 """
        cost_matrix = np.zeros((len(stored_positions), len(new_observations)))
        for i, (x1, y1) in enumerate(stored_positions):
            for j, (x2, y2) in enumerate(new_observations):
                cost_matrix[i, j] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = list(zip(row_ind, col_ind))
        return matches
    
    def update(self, obs_lists):
        """
        输入: obs_lists: List of lists, 每个子列表代表一个无人机的观测 (x, y, v, orientation) 或 None
        """
        self.timestamp += 1

        # 整理所有新的观测
        new_observations = []
        ol = []
        for obs_list in obs_lists:
            for obs in obs_list:
                if obs is not None:
                    new_observations.append(obs[:2])
                    ol.append(obs)

        # 如果没有新观测，则跳过
        if not new_observations:
            return

        # 获取当前的存储状态
        stored_positions = [
            (self.memory[idx]["x"], self.memory[idx]["y"])
            if self.memory[idx]["x"] is not None else (1e6, 1e6)  # 用一个极远的点表示未初始化
            for idx in range(self.num_enemies)
        ]

        # 使用匈牙利算法匹配
        matches = self._hungarian_match(stored_positions, new_observations)

        # 更新已有的状态
        for idx, obs_idx in matches:
            if obs_idx < len(new_observations):
                x, y = new_observations[obs_idx]
                obs = next(
                    (o for o in ol if o and o[0]==x and o[1]==y), None
                )
                if obs:
                    self.memory[idx].update({
                        "x": obs[0],
                        "y": obs[1],
                        # "v": obs[2],
                        # "orientation": obs[3],
                        "alive": True,
                        "last_seen": self.timestamp
                    })

        # 如果超过 5 个时间步未观测到，则判定为死亡
        for idx in range(self.num_enemies):
            if self.timestamp - self.memory[idx]["last_seen"] > self.dead_waiting:
                self.memory[idx]["alive"] = False

    def get_predictions(self):
        """
        返回所有敌方无人机的推测状态
        格式: [(alive, x, y, v, orientation), ...]
        """
        predictions = []
        for idx in range(self.num_enemies):
            state = self.memory[idx]
            if state["x"] is not None and state["y"] is not None:
                predictions.append((
                    1 if state["alive"] else 0,
                    state["x"],
                    state["y"],
                    state["v"] if state["v"] is not None else 0,
                    state["orientation"] if state["orientation"] is not None else 0
                ))
            else:
                predictions.append((0, 0, 0, 0, 0))
        return predictions



# Simple Test
if __name__ == "__main__":

    import time
    import env_utils
    from env_range_attack import BattleEnv

    num_agents = 5

    env = BattleEnv(red_agents=num_agents, blue_agents=num_agents, auto_record=False, developer_tools=True)
    state = env.reset()

    BB = BlackBox(num_agents)

    step = 0
    
    while True and step<1000:

        # 测试BlackBox
        ally_obs = env._search_enemies()
        BB.update(ally_obs)
        battle_predicted = BB.get_predictions()
        battle_real = env._get_enemies_position()

        print("Observations:", ally_obs)
        print("Battle predict:", battle_predicted)
        print("Battle real:", battle_real)




        # 生成随机动作（连续动作空间）
        actions = []

        for i in range(num_agents):
        # 一个飞机使用策略
            self_drone = env.drones[i]
            all_drones = env._get_all_drones()
            # self_drone = env._get_random_drone(1)
            actions.append(env_utils.control_strategy_Expert(self_drone, all_drones))

        for i in range(num_agents):
            # 一个飞机随机飞行
            actions.append(np.random.random(3) * 2 - 1)
            # actions.append([0, 0, 0])
        
        # 执行动作
        next_state, rewards, done, _ = env.step(actions)
        step += 1
        
        # 渲染
        # env.render(show_trail=True)
        time.sleep(1)
        
        # 检查终止条件
        if done:
            state = env.reset()
            BB.reset()
            
        # 控制台输出
        # print(f"当前奖励: {rewards}")