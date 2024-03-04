# CS370 - Introduction to Artificial Intelligence
# VIDEO SEARCH: Video indexing pipeline

## 1.0 Video library
Refer to /1-youtube-downloader/app-YouTube-Downloader-with-GUI.py
Download video and subtitles using Pytube and YouTube-Transcript-Api.

## 2.1 Preprocess the video
Refer to /1-youtube-downloader/app-extract-frames.py

## 2.2 Detecting objects
Refer to /2-detect-objects/download-custom-img && /2-detect-objects/Yolov8-custom

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
## 2.3 Embedding model
Refer to /3-embedding-model

1. ROI_bbox_compile.py
Use model to obtain bbox info and store in ROI.csv file
2. ROI_draw_bbox.py
Produce pictures with bbox info in ROI.csv file to validate prediction and store them in '/image-cropped-roi' folder.
3. ROI_crop.py

Function to crop images in original frames based on the bbox info of detected objects.
4. autoencoder-img-processing.py
Perform image preprocessing to prepare image dimension for the model input. This includes resizing the image to (128, 128, 3), convert the image to grayscale, convert the image to a numpy array like MNIST dataset, and normalize.
This produces "./mnistlikedataset224x1.npz" file containing the image array.
5. autoencoder-main.py
This is the main training file containing the custom convolutional autoencoder architecture.
6. autoencoder-img-reconstruction.ipynb
Demonstration of the model's performance through image reconstruction.
7. autoencoder-similarity-search.ipynb
Demonstration of cosine similarity search to query top 5 images based on an embedding.
[](cs370-tn268-introduction-to-ai-assignments\assignment-2-video-search\3-embedding-model\autoencoder-performance\similarity-search-output-2.pngcs370-tn268-introduction-to-ai-assignments\assignment-2-video-search\3-embedding-model\autoencoder-performance\similarity-search-output-2.png)

## Indexing the embeddings
Use docker compose to bring up two docker containers, your application container with the dev environment (you must have done this in Step 1) and a second container with postgres.
