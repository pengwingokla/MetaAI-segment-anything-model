import csv
import math
import random
import torch
import cv2
import numpy as np
from ultralytics import YOLO

""" This code store all the center points of detected objects in the video"""

# OUTPUT CSV -------------------------------
CSV_PATH  = '2-ODmodel/object_positions.csv'
# ------------------------------------------

CLASS_DIR = '2-ODmodel/VisDrone.txt'
MODEL_DIR = '2-ODmodel/result/100_epochs_3/weights/best.pt'
INPUT_VID = 'DSVIDEO-drone/1-cyclist-vehicle.mp4'
OUT_VIDEO = '2-ODmodel/VID3-3-kalman.avi'
FPS = 10

def tensor_to_list(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.tolist()
    else:
        raise TypeError("Input must be a PyTorch tensor")
    
# Open CSV file in write mode
with open(CSV_PATH, 'w', newline='') as csvfile:
    # Create CSV writer object
    csvwriter = csv.writer(csvfile)

    # Write the header row
    csvwriter.writerow(['Frame', 'Object ID', 'Center X', 'Center Y'])

    my_file = open(CLASS_DIR, "r")
    data = my_file.read()
    class_list = data.split("\n") # replacing end splitting the text when newline ('\n') is seen.
    my_file.close()


    # Generate random colors for class list
    detection_colors = []
    for i in range(len(class_list)):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        detection_colors.append((b, g, r))

    # load pretrained model
    model = YOLO(MODEL_DIR)

    cap = cv2.VideoCapture(INPUT_VID)

    frame_wid = int(cap.get(3)) 
    frame_hyt = int(cap.get(4)) 

    dimension = (frame_wid, frame_hyt) 

    # record the video with predicted bbox
    out = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*'MJPG'), FPS, dimension) 

    if not cap.isOpened():
        print("Cannot open camera")
        exit()


    # Object Tracker
    tracking_object = {}
    prev_cxcy = []
    frame_num = 0

    while True:
        ret, frame = cap.read()
        frame_num += 1

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # PREDICT ON IMAGE
        detect_params = model.predict(source=[frame], conf=0.45, save=True)
        # -- Convert tensor array
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

        # -- Add center points of detected obj
        for pt1, pt2 in zip(curr_cxcy, prev_cxcy):
            distance = math.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1])
            if distance < 10:
                for idx, pt1 in enumerate(curr_cxcy):
                    if (idx==0) or (idx in track_obID):
                        obID = idx
                        tracking_object[obID] = pt1
                        # Write object center coordinates to CSV
                        csvwriter.writerow([int(frame_num), obID, pt1[0], pt1[1]])
                    else:
                        obID = max(track_obID) + 1
                        tracking_object[obID] = pt1
                        # Write object center coordinates to CSV
                        csvwriter.writerow([int(frame_num), obID, pt1[0], pt1[1]])
                    cv2.circle(img=frame, center=pt1, radius=1, color=(255,0,0), thickness=4)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img=frame, text=str(obID)+" "+str(pt1), org=(pt1[0],pt1[1]-7), fontFace=font, fontScale=0.5, color=(255, 255, 255), thickness=1,)

        track_obID = list(tracking_object.keys())

        
        prev_cxcy = curr_cxcy.copy()
        
        # DRAW BOUNDING BOXES, CLASS, CONF
        if len(cls) != 0:
            for i in range(len(detect_params[0])):
                clsID = cls[i]
                bb = bbox[i]
                cf = conf[i]
                cv2.rectangle(frame,(int(bb[0]), int(bb[1])),(int(bb[2]), int(bb[3])), detection_colors[int(clsID)],3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img=frame,text=class_list[int(clsID)] + " " + str(round(cf, 3)),org=(int(bb[0]), int(bb[1]) - 10),fontFace=font,fontScale=0.5,color=(255, 255, 255),thickness=1)

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
