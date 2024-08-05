import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

# 解析XML文件
def parse_tripinfo(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    time_losses = []
    for tripinfo in root.findall('tripinfo'):
        time_loss = float(tripinfo.get('timeLoss'))
        time_losses.append(time_loss)
    return time_losses

# 绘制直方图
def plot_histogram(time_losses):
    average_time_loss = np.mean(time_losses)
    
    plt.figure(figsize=(10, 6))
    plt.hist(time_losses, bins=20, edgecolor='black')
    plt.axvline(average_time_loss, color='red', linestyle='dashed', linewidth=1)
    plt.text(average_time_loss + 2, plt.ylim()[1] * 0.9, f'Average: {average_time_loss:.2f}', color='red')
    plt.title('Histogram of Time Losses (Full Range)')
    plt.xlabel('Time Loss')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# 主函数
def main():
    file_path = 'sumo_rl/nets/two_intersections/2024-07-24-10-23-14tripinfo.xml'  # 替换为你的文件路径
    time_losses = parse_tripinfo(file_path)
    plot_histogram(time_losses)

if __name__ == '__main__':
    main()