import json
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

# 读取 jsonl 文件
log_path = r"uniform\18_curr_test\reward_part_3.jsonl"

# 加载所有记录
data = []
with open(log_path, "r") as f:
    for line in f:
        data.append(json.loads(line))

# 提取每帧中的奖励分项
reward_records = []
for entry in data:
    epoch = entry["epoch"]
    for frame in entry["frames"]:
        frame_index = frame["frame_index"]
        for reward in frame["rewards"]:
            reward_copy = reward.copy()
            reward_copy["epoch"] = epoch
            reward_copy["frame_index"] = frame_index
            reward_records.append(reward_copy)

# 转换为 DataFrame
df = pd.DataFrame(reward_records)

# 选择一个 epoch 和一个智能体 ID 进行可视化
selected_epoch = 300
selected_id = 0
df_filtered = df[(df["epoch"] == selected_epoch) & (df["id"] == selected_id)]

# 筛选 reward 分项字段（排除 meta 字段）
exclude_cols = {"id", "team", "alive", "total", "epoch", "frame_index"}
reward_keys = [col for col in df_filtered.columns if col not in exclude_cols]

# 可视化：绘制 reward 分项的堆叠柱状图
plt.figure(figsize=(14, 6))
bottom = [0] * len(df_filtered)
x = df_filtered["frame_index"]

for key in reward_keys:
    plt.bar(x, df_filtered[key], bottom=bottom, label=key)
    bottom = [i + j for i, j in zip(bottom, df_filtered[key])]

plt.title(f"Reward Breakdown over Time (Epoch {selected_epoch}, Agent ID {selected_id})")
plt.xlabel("Frame Index")
plt.ylabel("Reward Value")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.grid(True)
plt.show()
