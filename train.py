from env import BattleEnv
from maddpg import MADDPG, train





# 运行训练
if __name__ == "__main__":

    # task_series = "F_commu"
    task_code = "acc_action_test02"

    env = BattleEnv(red_agents=2, blue_agents=2, auto_record=True)
    rewards = train(env, episodes=20, is_render=False, task_code=task_code)

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