# Cyclist and Car Detection on Drone Video using Kalman Filter

---
## 🚵‍♀️ Task 1: TF2 Docker Container
For testing the docker containers, sample codes are added into main.py files. Remove them before use!

### Run Python files

Put all your code inside the src folder and change `"python main.py"` line inside docker-compose file with your main python folder or script. src folder will directly be copied inside the container.
```
docker build .
docker-compose build
docker-compose up
```
### Run Jupyter-Notebok

Go to the docker compose folder, change `"python main.py"` with `"jupyter-notebook --ip 0.0.0.0 --port 8000"`. Then, follow the instructions:
```
docker build .
docker-compose build
docker-compose up
```
After that, you should see a prompt saying that server is online at http://127.0.0.1:8000/?token=c1b1f0... Use that link to access your notebook. If you want to use your notebook in your local network, replace 127.0.0.1 with your computer's ip address. Then, you should be able to access it in your local network.

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

- `kalman.py` is the main code of the Kalman Filter specifying the params, prediction step and update step. This code outputs `kalman-output.csv` that is used to plot the predicted center points on the frames.
- `main_predict_video.py` applies Object Detection Model and Kalman Filter on the video and produces a video with bounding boxes of the detected cars/cyclist and the kalman center points to track the vehicle.

