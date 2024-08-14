import pandas as pd
import matplotlib.pyplot as plt

def plot_timeloss_histogram(file_path):
    # Load the data from the CSV file
    data = pd.read_csv(file_path)

    # Extract the 'Final TimeLoss' column
    time_loss = data['Final TimeLoss']

    # Calculate the mean of the 'Final TimeLoss'
    mean_time_loss = time_loss.mean()

    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(time_loss, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(mean_time_loss, color='red', linestyle='dashed', linewidth=1)
    plt.text(mean_time_loss, plt.ylim()[1] * 0.9, f'Mean: {mean_time_loss:.2f}', color='red')
    plt.title('Distribution of TimeLoss')
    plt.xlabel('TimeLoss')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Example usage
file_path = 'final_timeloss_data.csv'  # Replace with your file path
plot_timeloss_histogram(file_path)
