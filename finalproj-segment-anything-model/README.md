# Segment Anything Model

## 🚵‍♀️ Milestone 1: Torch Docker Container
For testing the docker containers, sample codes are added into main.py files.

```
docker build -t working-sam .
docker-compose up --build
```
For Jupyter notebooks, after building the container you should see a prompt saying that server is online at http://127.0.0.1:8888/tree/notebooks... Use that link to access your notebook. If you want to use your notebook in your local network, replace 127.0.0.1 with your computer's ip address. Then, you should be able to access it in your local network.

The `requirements.txt` ensures the following are installed
```
torch==2.2.2
torchvision==0.17.2
```

## 🚵‍♀️ Milestone 2: Replicate SAM Implementation

The `sam-implementation.ipynb` notebook shows how to use segment satellite imagery using the Segment Anything Model (SAM). Use the this Colab link to view the interactive maps

The content inside notebook is to big to display so please refer to this Colab.

[![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IZ-54GI-5cp5oTfhc8_NKHkIZRp8Yw_A?usp=sharing)


## 🚵‍♀️ Milestone 3: Finetune the SAM model for the sidewalks dataset
<br>`stream.ipynb` all training processes
<br>`predict.ipynb` demonstrates the prediction on Google Earth screenshot

## 🚵‍♀️ Milesone 4: Hugging Face App and Video Demo

[Hugging Face](https://huggingface.co/spaces/chloecodes/sam)

[Video Presentation](https://drive.google.com/file/d/1pfNURxCghOVTSUYtp3h53hsBx2Jq6jhZ/view?usp=sharing)