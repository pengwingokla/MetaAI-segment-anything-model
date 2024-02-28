from ultralytics import YOLO
from matplotlib import pyplot as plt
import torch
import csv
import os

image = 'assignment-2-video-search\\3-embedding-model\\image-pred\\video-1\\001_4255.jpg'
model_path  = 'assignment-2-video-search\\3-embedding-model\\best.pt'
# root_DATASET = 'assignment-2-video-search\\1-youtube-downloader\\DATASET-FRAMES'
# save_path   = 'assignment-2-video-search\\3-embedding-model\\image-pred-bbox'

root_DATASET = 'assignment-2-video-search\\3-embedding-model\\image-pred'
save_path = 'assignment-2-video-search\\3-embedding-model\\image-pred-bbox'

model = YOLO(model_path)

print(">>>>>>> PROCESSING...", image)
if image.endswith('.jpg'):

    # Predict on the current image file
    pred_imgs = model.predict(image, conf=0.3)
    image_filename = os.path.basename(image)[:-4]
    
    print(pred_imgs)

    # Iterate over each prediction, plot, and save
    for i, pred_img in enumerate(pred_imgs):
        result_array = pred_img.plot()
        plt.imshow(result_array)
        plt.title(f"pred_{image_filename}") 
        plt.savefig(f'{save_path}\\pred_{image_filename}.jpg', dpi=300)

    # print(">>>>>>> COMPLETED", image," with bounding box prediction is stored!")
        
