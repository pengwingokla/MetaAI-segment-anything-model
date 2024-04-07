# Cyclist and Car Detection on Drone Video using Kalman Filter

---
## 🚵‍♀️ Task 1: TF2 Docker Container
For testing the docker containers, sample codes are added into main.py files. Remove them before use!

### Run Python files

```
docker build -t dronetf .
docker run dronetf
```

---
## 🚵‍♀️ Task 2: Object Detection
See object detection output videos at this [Google Drive](https://drive.google.com/drive/folders/1rpOvINEG87zVAyD-nCcOF6t6y0vgJ7T0?usp=sharing)

Check out the model evaluation here at this [Google Drive](https://drive.google.com/drive/folders/1RQNx_mMQAHIdYgOIAWFEJ1oC_wP86Vad?usp=sharing)

See folder `2-ODmodel`
- `dataset-filter.py` and `dataset-modify.py` are to modify the original VisDrone2019 dataset to filter out the bicycle and cars classes
- `eda.py` is to perform exploratory data analysis on the class instances in the filtered dataset. EDA shows that the dataset is imbalanced, causing the model to detect cars more often than cyclist.
- `main_objectdetect_train.ipynb` is the training of YOLOv8 on the bicycle and car dataset. 
- `test_predict_frames.py` outputs a folder of all the frames with boudning boxes drawn on the detected object. This step is done before using the model on the actual video in case the model requires any finetuning.


---
## 🚵‍♀️ Task 3: Kalman Filter
See Kalman car tracking videos at this [Google Drive](https:/drive.google.com/file/d/1JR6Qwm_zHE3128PMVOrdfBHu0sTBgJdJ/view?usp=sharing)

See folder `3-Kalman`
- `kf_internal.py` and `book_plot.py` are part of the filterpy library.
- `plot.ipynb` demonstrates how kalman prediction csv is produced. 
- `kalman.py` is the main code of the Kalman Filter specifying the params, prediction step and update step. This code outputs `kalman-output.csv` that is used to plot the predicted center points on the frames.
- `main_predict_video.py` applies Object Detection Model and Kalman Filter on the video and produces a video with bounding boxes of the detected cars/cyclist and the kalman center points to track the vehicle.

