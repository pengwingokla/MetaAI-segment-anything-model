import os
from PIL import Image
import csv

# Root directory where the images are located
crop_img_dir = "assignment-2-video-search\\3-embedding-model\\image-cropped-rois"
csv_file_dir = "assignment-2-video-search\\3-embedding-model\\ROI_data.csv"
output_dir   = "assignment-2-video-search\\3-embedding-model\\model-input-dataset"

def classify_to_folder(csv_file_dir, crop_img_dir):
    """This function classify cropped ROI images into folder of its label"""
    # for i in range(89):
    #     directory_name = str(i)
    #     directory_path = os.path.join(output_dir, directory_name)
    #     os.makedirs(directory_path)

    with open(csv_file_dir, 'r') as file:
        csv_reader = csv.DictReader(file)
        for imagefile, row in zip(sorted(os.listdir(crop_img_dir)), csv_reader):
            label    = str(int(float(row['label'])))
            label_folder = os.path.join(output_dir, label)
            image_path   = os.path.join(crop_img_dir , imagefile)
            image = Image.open(image_path)
            image_copy = image.copy()
            image_copy.save(f"{label_folder}\\{imagefile}")

def folder_filecount(base_directory):
    folder_file_count = {}

    for root, dirs, files in os.walk(base_directory):
        label = os.path.relpath(root, base_directory)
        file_count = len(files)
        folder_file_count[label] = file_count

    return folder_file_count

# classify_to_folder(csv_file_dir, crop_img_dir)

# Returns the count of ROI images in each class
for folder, count in folder_filecount(output_dir).items():
    if count == 0 and folder != ("."): # Delete empty folders
        folder_path = os.path.join(output_dir, folder)
        os.rmdir(folder_path)
    else:
        print(f"Label: {folder}, Number of Image Files: {count}")
