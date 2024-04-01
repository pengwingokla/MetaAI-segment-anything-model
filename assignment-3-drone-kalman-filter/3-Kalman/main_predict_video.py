import random
import torch
import cv2
import csv
import numpy as np
import math
from ultralytics import YOLO

CLASS_DIR = '2-ODmodel/VisDrone.txt'
MODEL_DIR = '2-ODmodel/result/100_epochs_3/weights/best.pt'
INPUT_VID = 'DSVIDEO-drone/3-drone-tracking.mp4'
OUT_VIDEO = '2-ODmodel/VID3-3-kalman.avi'
FPS = 10

def tensor_to_list(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.tolist()
    else:
        raise TypeError("Input must be a PyTorch tensor")
    
THRES_FRAME = 20
def kalman_pred(frame_num):
    coordinates = []
    with open('2-ODmodel/kalman_output.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header row
        for row in csvreader:
            if int(row[0]) <= frame_num and int(row[0]) >= frame_num - THRES_FRAME:
                coordinates.append((float(row[1]), float(row[2])))
        return coordinates

object_cls = open(CLASS_DIR, "r")
data = object_cls.read()
class_list = data.split("\n")
object_cls.close()

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Load pretrained model
model = YOLO(MODEL_DIR)


# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture(INPUT_VID)

frame_wid = int(cap.get(3)) 
frame_hyt = int(cap.get(4)) 

dimension = (frame_wid, frame_hyt) 

# record the video with predicted bbox
out = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*'MJPG'), FPS, dimension) 

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    # if frame is read correctly ret is True

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # PREDICT ON IMAGE
    detect_params = model.predict(source=[frame], conf=0.45, save=True)
    # Convert tensor array
    for pred in detect_params:
        cls = tensor_to_list(pred.boxes.cls)
        bbox = tensor_to_list(pred.boxes.xyxy)
        conf = tensor_to_list(pred.boxes.conf)

    # TRACK DETECTED OBJECT CURR & PREV POSITIONS
    curr_cxcy = []
    for xy in bbox:
        cx = int((xy[0] + xy[2]) / 2)
        cy = int((xy[1] + xy[3]) / 2)
        center_pt = (cx, cy)
        curr_cxcy.append(center_pt)

    print("CURR: ", curr_cxcy)
    print("PREV: ", prev_cxcy)
    # -- DRAW CENTER POINTS OF DETECTED OBJECTS
    for pt1, pt2 in zip(curr_cxcy, prev_cxcy):
        
        print('distance:',pt1, pt2)
        distance = math.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1])
        
        print('distance:',distance)
        if distance < 10:
            for idx, pt1 in enumerate(curr_cxcy):
                print('current idx:',idx)
                if (idx==0) or (idx in track_obID):
                    obID = idx
                    tracking_object[obID] = pt1
                else:
                    obID = max(track_obID) + 1
                    tracking_object[obID] = pt1

                cv2.circle(img=frame, center=pt1, radius=1, color=(255,0,0), thickness=4)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img=frame, text=str(obID)+" "+str(pt1), org=(pt1[0],pt1[1]-7), fontFace=font, fontScale=0.5, color=(255, 255, 255), thickness=1,)
      
    # DRAW KALMAN PRED
    kalman_xy = kalman_pred(frame_num)
    print('KALMAN xy:', kalman_xy)
    for xy in kalman_xy:
        x = int(xy[0]) ; y = int(xy[1])
        cv2.circle(img=frame, center=(x,y), radius=1, color=(255,204,255), thickness=4)

    track_obID = list(tracking_object.keys())

    prev_cxcy = curr_cxcy.copy()
    print("track_obID: ",track_obID)

    # DRAW BOUNDING BOXES, CLASS, CONF
    if len(cls) != 0:
        for i in range(len(detect_params[0])):

            clsID = cls[i]
            bb = bbox[i]
            cf = conf[i]

            # BBOX RECTANGLE
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            # BBOX CLASS LABEL TEXT AND CONF
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

    # Write the frame
    out.write(frame)

    # Display the resulting frame
    cv2.imshow("ObjectDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
out.release() 
cv2.destroyAllWindows()