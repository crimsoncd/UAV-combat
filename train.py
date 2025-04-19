from env import BattleEnv
from maddpg import MADDPG, train, train_Revised





# 运行训练
if __name__ == "__main__":

    # task_series = "F_commu"7
    task_code = "12_Revised_2nd_CTDE_cuda"

    env = BattleEnv(red_agents=2, blue_agents=2, auto_record=True)
    rewards = train_Revised(env, episodes=3000, is_render=False, task_code=task_code)

    exit(0)
    
    # 训练后测试
    state = env.reset()
    agent = MADDPG(env)
    while True:
        actions = agent.act(state)  # 关闭探索噪声
        next_state, rewards, done, _ = env.step(actions)
        env.render()
        if done:
            state = env.reset()
        else:
            state = next_state