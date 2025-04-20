import matplotlib.pyplot as plt
import re



# 初始化数据容器
episodes, rewards, alosses, closses = [], [], [], []


log_file = r"uniform\12_Revised_2nd_CTDE_cuda\log.txt"
pic_file = log_file[:-4] + ".png"

# 解析日志数据（若从文件读取，替换为 with open("log.txt") as f: ...）
with open(log_file, "r") as lf:
    for line in lf:
        # 使用正则表达式提取数值
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        episodes.append(int(matches[0]))
        rewards.append(float(matches[1]))
        alosses.append(float(matches[4]))
        closses.append(float(matches[5]))

# 创建三个并排子图
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# 绘制 Reward 曲线
ax1.plot(episodes, rewards, "b-", linewidth=1)
ax1.set_title("Reward per Episode")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Reward")
ax1.grid(True, alpha=0.3)

# 绘制 Actor Loss 曲线
ax2.plot(episodes[10:], alosses[10:], "r-", linewidth=1)
ax2.set_title("Actor Loss per Episode")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Actor Loss")
ax2.grid(True, alpha=0.3)

# 绘制 Critic Loss 曲线
ax3.plot(episodes[10:], closses[10:], "g-", linewidth=1)
ax3.set_title("Critic Loss per Episode")
ax3.set_xlabel("Episode")
ax3.set_ylabel("Critic Loss")
ax3.grid(True, alpha=0.3)

# 优化布局
plt.tight_layout()
plt.savefig(pic_file)
plt.show()