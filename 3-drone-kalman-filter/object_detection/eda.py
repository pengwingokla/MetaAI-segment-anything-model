"""" Exploratory Data Analysis """
          
import os
import matplotlib.pyplot as plt
import numpy as np

def count_train(folder_path, output_folder):
    # Dictionary to store counts for each starting number
    count_by_number = {}

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as file:
            for line in file:
                # Extract the first number from the line
                first_number = line.strip().split()[0]

                # Update count for the first number
                count_by_number[first_number] = count_by_number.get(first_number, 0) + 1

    # Sort the counts based on the first number
    sorted_counts = sorted(count_by_number.items(), key=lambda x: int(x[0]))

    # Extract keys and values from sorted counts
    numbers, counts = zip(*sorted_counts)

    # Plot the bar graph
    plt.bar(numbers, counts, color='royalblue') # <--- CHANGE HERE
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Custom Train Dataset: Count of Class Instances')  # <--- CHANGE HERE
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the plot as a picture in the output folder
    output_path = os.path.join(output_folder, 'custom_class_count_train_2.png') # <--- CHANGE HERE
    plt.savefig(output_path)
    plt.close()

    print(f"Bar graph saved as {output_path}")

# Example usage                                                                               V--- CHANGE HERE
input_folder = "D:\Drone-Kalman-Filters\drone-kalman-filters\\2-obj-detection-model\datasets-custom\VisDrone\VisDrone2019-DET-train\labels"
output_folder = "D:\Drone-Kalman-Filters\drone-kalman-filters\\2-obj-detection-model\\eda-graphs"
count_train(input_folder, output_folder)

# ---------------------------------------------------------------------------------

def count_accumulated(input_folders, output_folder):
    # Dictionary to store counts for each starting number
    count_by_number = {}

    # Iterate over input folders
    for folder_path in input_folders:
        # Iterate over files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r') as file:
                for line in file:
                    # Extract the first number from the line
                    first_number = line.strip().split()[0]

                    # Update count for the first number
                    count_by_number[first_number] = count_by_number.get(first_number, 0) + 1

    # Plot the bar graph
    plt.bar(count_by_number.keys(), count_by_number.values())
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Accumulated Count of Class Instances')

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the plot as a picture in the output folder
    output_path = os.path.join(output_folder, 'bar_graph_class_count_accumulated.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Bar graph saved as {output_path}")

# Example usage
input_folders = ["D:\Drone-Kalman-Filters\drone-kalman-filters\\2-obj-detection-model\datasets\VisDrone\VisDrone2019-DET-train\labels", 
                 "D:\Drone-Kalman-Filters\drone-kalman-filters\\2-obj-detection-model\datasets\VisDrone\VisDrone2019-DET-val\labels",]
output_folder = "D:\Drone-Kalman-Filters\drone-kalman-filters\\2-obj-detection-model\\eda-graphs"
# count_accumulated(input_folders, output_folder)

# ---------------------------------------------------------------------------------

def count_distribution(input_folders, output_folder):
    # Dictionary to store counts for each starting number and folder
    count_by_number = {}

    # Iterate over input folders
    for folder_idx, folder_path in enumerate(input_folders):
        # Assign labels "Train" and "Val" to the folders
        folder_label = "Train" if folder_idx == 0 else "Val"

        # Iterate over files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # Open each file for reading
            with open(file_path, 'r') as file:
                # Iterate over each line in the file
                for line in file:
                    # Extract the first number from the line
                    first_number = line.strip().split()[0]

                    # Update count for the first number and folder
                    key = (first_number, folder_label)
                    count_by_number[key] = count_by_number.get(key, 0) + 1

    # Extract unique numbers and folders
    numbers = sorted(set(number for number, _ in count_by_number.keys()))

    # Prepare data for plotting
    counts_by_folder = np.zeros((len(numbers), 2))
    for i, number in enumerate(numbers):
        for folder_label in ["Train", "Val"]:
            key = (number, folder_label)
            counts_by_folder[i, 0 if folder_label == "Train" else 1] = count_by_number.get(key, 0)

    # Plot the bar graph
    width = 0.4  # Width of each bar
    ind = np.arange(len(numbers))  # Indices for each bar group

    # Set the color for the bars
    
    plt.bar(ind, counts_by_folder[:, 0], width=width, label="Train", color='black')
    plt.bar(ind + width, counts_by_folder[:, 1], width=width, label="Val", color='dodgerblue')

    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Original Dataset Class Instance Distribution')
    plt.xticks(ind + width / 2, numbers)
    plt.legend()

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the plot as a picture in the output folder
    output_path = os.path.join(output_folder, 'original_class_count_distribution.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Bar graph saved as {output_path}")

# Example usage
input_folders = ["D:\Drone-Kalman-Filters\drone-kalman-filters\\2-obj-detection-model\datasets\VisDrone\VisDrone2019-DET-train\labels", 
                 "D:\Drone-Kalman-Filters\drone-kalman-filters\\2-obj-detection-model\datasets\VisDrone\VisDrone2019-DET-val\labels"]
output_folder = "D:\Drone-Kalman-Filters\drone-kalman-filters\\2-obj-detection-model\\eda-graphs"
# count_distribution(input_folders, output_folder)