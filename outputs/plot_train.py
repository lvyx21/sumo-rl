import os
import json
import matplotlib.pyplot as plt
import pandas as pd



# 绘制训练损失图
def plot_training_loss(tfile_path):
    data = pd.read_csv(file_path)
    step=data['step']
    system_total_waiting_time=data['system_total_waiting_time']
 

    plt.figure(figsize=(10, 5))
    plt.plot(step, system_total_waiting_time,  alpha=0.5)
    plt.scatter(step, system_total_waiting_time, s=1, color='r')
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Training Performance")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 8000)  # 设置横坐标范围从0到80000
    plt.show()

file_path = 'outputs/two_intersections/ppo_conn0_ep2.csv'  # Replace with your file path
plot_training_loss(file_path)