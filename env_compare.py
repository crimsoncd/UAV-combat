from env_range_attack import BattleEnv
from config import *

class DualBattleEnv():

    def __init__(self, red_agents=5, blue_agents=5, auto_record=True, developer_tools=False,
                 margin_crash=True, collision_crash=True):
        
        self.red_agents = red_agents
        self.blue_agents = blue_agents

        self.BattleA = BattleEnv(red_agents, blue_agents, auto_record, developer_tools, margin_crash, collision_crash)
        self.BattleB = BattleEnv(red_agents, blue_agents, auto_record, developer_tools, margin_crash, collision_crash)
        

    def reset_all(self):

        self.BattleA.reset()
        self.BattleB.reset()

        for i in range(self.red_agents + self.blue_agents):
            x, y = np.random.uniform(50, MAP_SIZE_0 - 50), np.random.uniform(50, MAP_SIZE_1 - 50)
            self.BattleA.drones[i].x = x
            self.BattleA.drones[i].y = y
            self.BattleB.drones[i].x = x
            self.BattleB.drones[i].y = y
