from env import BattleEnv
import json
import time

def film_play(record_file, play_epoch=[], play_speed=10):

    play_interval = 0.1 / play_speed

    play_data = []
    line_idx = 0
    with open(record_file, 'r') as f:
        for line in f:
            if len(play_epoch)==0 or line_idx in play_epoch:
                play_data.append(json.loads(line)["frames"])
            line_idx += 1
    
    # 加载实例
    obj_states = play_data[0][0]["objects"]
    red_num,blue_num = 0,0
    for obj in obj_states:
        if obj['type']=='Drone' and obj['state']["team"]=="red":
            red_num += 1
        elif obj['type']=='Drone' and obj['state']["team"]=="blue":
            blue_num += 1

    env = BattleEnv(red_agents=red_num, blue_agents=blue_num, auto_record=False, developer_tools=True)
    for epoch_data in play_data:
        env.reset()
        for frame_data in epoch_data:
            env.set_frame_data(frame_data['objects'])
            env.render(show_trail=True)
            time.sleep(play_interval)


if __name__=="__main__":

    play_file = r"uniform\23_Mix_Expert_MADDPG_AZ_e\record_part_17.jsonl"
    film_play(play_file, play_epoch=[], play_speed=10)