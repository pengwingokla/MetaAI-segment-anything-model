import random
import torch
import cv2
import numpy as np
from ultralytics import YOLO

CLASS_DIR = '2-ODmodel/VisDone.txt'
MODEL_DIR = '2-ODmodel/result/10_epochs_12/weights/best.pt'
INPUT_VID = 'DSVIDEO-drone/2-cyclist-vehicle.mp4'
OUT_VIDEO = 'VID2.avi'

def tensor_to_list(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.tolist()
    else:
        raise TypeError("Input must be a PyTorch tensor")
    
# opening the file in read mode
my_file = open(CLASS_DIR, "r")
data = my_file.read()
# replacing end splitting the text when newline is seen.
class_list = data.split("\n")
my_file.close()

# print(class_list)

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# load pretrained model
model = YOLO(MODEL_DIR)


# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture(INPUT_VID)

frame_wid = int(cap.get(3)) 
frame_hyt = int(cap.get(4)) 

dimension = (frame_wid, frame_hyt) 

out = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*'MJPG'), 10, dimension) 

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #  resize the frame | small frame optimise the run
    # frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.45, save=True)

    # Convert tensor array to numpy
    for pred in detect_params:
        cls = tensor_to_list(pred.boxes.cls)
        bbox = tensor_to_list(pred.boxes.xyxy)
        conf = tensor_to_list(pred.boxes.conf)
    

    if len(cls) != 0:
        for i in range(len(detect_params[0])):

            clsID = cls[i]
            bb = bbox[i]
            cf = conf[i]

            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            # Display class name and confidence
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img=frame,
                text=class_list[int(clsID)] + " " + str(round(cf, 3)),
                org=(int(bb[0]), int(bb[1]) - 10),
                fontFace=font,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=1,
            )

    # write the frame
    out.write(frame)

    # Display the resulting frame
    cv2.imshow("ObjectDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cap.release()
out.release() 
cv2.destroyAllWindows()