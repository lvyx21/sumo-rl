import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = '/home/lvyx/Documents/sumo-rl-main/final_timeloss_data.csv'  # 请将'你的文件路径'替换为实际文件路径
data = pd.read_csv(file_path)

# 过滤出所有vehicle id是flow12.xx的车
filtered_data = data[data['Vehicle ID'].str.startswith('flow12.')]

# 提取timeloss值
timeloss_values = filtered_data['Final TimeLoss']

# 计算timeloss平均值
mean_timeloss = np.mean(timeloss_values)

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(timeloss_values, bins=20, color='blue', edgecolor='black')
plt.axvline(mean_timeloss, color='red', linestyle='dashed', linewidth=1)
plt.text(mean_timeloss + 1, max(np.histogram(timeloss_values, bins=20)[0])/2, f'Mean: {mean_timeloss:.2f}', color='red')
plt.xlabel('TimeLoss')
plt.ylabel('Frequency')
plt.title('TimeLoss Distribution for Vehicles with ID flow12.xx')
plt.grid(True)
plt.show()

print(f'Mean TimeLoss: {mean_timeloss}')
