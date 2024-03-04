# CS370 - Introduction to Artificial Intelligence
# VIDEO SEARCH: Video indexing pipeline

## 1.0 Video library
Refer to /1-youtube-downloader/app-YouTube-Downloader-with-GUI.py

## 2.1 Preprocess the video
Refer to /1-youtube-downloader/app-extract-frames.py
1. Download COCO Dataset - Common Objects in Context 
https://cocodataset.org/#download
2. Add images and class labesl for YOLOv8 to train
2.1 Download building and plant images using simple-image-downloader and labelImg
2.2 Add these images and labels to COCO current dataset
3. Split dataset into train and val folders
4. Configure the model on its yaml file 
assignment-2-video-search\2-detect-objects\Yolov8-custom\coco-custom.yaml
5. Train YOLO on new custom dataset
assignment-2-video-search\2-detect-objects\Yolov8-custom\YOLO_V8_main.ipynb
6. Check model's performance in the '/results' folder
7. Get YOLO best performance model in '/results/100_epochs-/weights/best.pt'

## 2.2 Detecting objects
Refer to /2-detect-objects/download-custom-img && /2-detect-objects/Yolov8-custom

## 2.3 Embedding model
Refer to /3-embedding-model

## Indexing the embeddings
Use docker compose to bring up two docker containers, your application container with the dev environment (you must have done this in Step 1) and a second container with postgres.
