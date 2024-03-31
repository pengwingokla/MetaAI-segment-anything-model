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

---
## 🚵‍♀️ Task 3: Kalman Filter
See Kalman car tracking videos at this [Google Drive](https://drive.google.com/file/d/1JR6Qwm_zHE3128PMVOrdfBHu0sTBgJdJ/view?usp=sharing)
