from ultralytics import YOLO
from matplotlib import pyplot as plt
from IPython.display import Image
import torch
import yaml
import os

import locale
locale.getpreferredencoding = lambda: "UTF-8"


MODEL_DIR = '2-ODmodel\\result\\10_epochs\weights\\best.pt'
# MODEL_DIR = '2-ODmodel\\result/100_epochs_2/weights/best.pt'
# FOLDER_IN = 'DSFRAMES\FRAMES-3-cyclist-and-vehicle-tracking-1'
# FOLDER_OT = '2-ODmodel\prediction-output-3-3' 
FOLDER_IN = 'DSFRAMES\FRAMES-2-cyclist-and-vehicle-tracking-2'
FOLDER_OT = '2-ODmodel\prediction-output-2-2' 
# FOLDER_IN = 'DSFRAMES\FRAMES-1-drone-tracking-video'
# FOLDER_OT = '2-ODmodel\prediction-output-1-3' 
os.makedirs(FOLDER_OT, exist_ok=True)  


model = YOLO(MODEL_DIR)


image_files = [file for file in os.listdir(FOLDER_IN) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]


for image_file in image_files:
    # Predict on the current image file
    image_path = os.path.join(FOLDER_IN, image_file)
    pred_imgs = model.predict(image_path, conf=0.3)

    # Save the predicted image
    result_array = pred_imgs[0].plot()
    output_path = os.path.join(FOLDER_OT, image_file)
    plt.imsave(output_path, result_array)

    print(f"Saved: {output_path}")

    # Close the plot to avoid displaying it
    plt.close()

print("All images saved successfully.")
