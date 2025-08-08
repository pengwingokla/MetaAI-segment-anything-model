import math
import random
import torch
import cv2
import csv
import numpy as np
from ultralytics import YOLO

class ObjectTracker:
    def __init__(self, kalman_csv, class_dir, model_dir, input_vid, out_video, fps=10, thres_frame=10):
        self.kalman_csv = kalman_csv
        self.class_dir = class_dir
        self.model_dir = model_dir
        self.input_vid = input_vid
        self.out_video = out_video
        self.fps = fps
        self.thres_frame = thres_frame

        # Load class list
        object_cls = open(self.class_dir, "r")
        data = object_cls.read()
        self.class_list = data.split("\n")
        object_cls.close()

        # Generate random colors for class list
        self.detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
                                  for _ in range(len(self.class_list))]

        # Load pretrained model
        self.model = YOLO(self.model_dir)

    def tensor_to_list(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.tolist()
        else:
            raise TypeError("Input must be a PyTorch tensor")

    def kalman_pred_coordinate(self, frame_num):
        coordinates = []
        with open(self.kalman_csv, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Skip header row
            for row in csvreader:
                if int(row[0]) <= frame_num and int(row[0]) >= frame_num - self.thres_frame:
                    coordinates.append((float(row[1]), float(row[2])))
            return coordinates

    def track_objects(self):
        cap = cv2.VideoCapture(self.input_vid)
        frame_wid = int(cap.get(3))
        frame_hyt = int(cap.get(4))
        dimension = (frame_wid, frame_hyt)

        # Record the video with predicted bbox
        out = cv2.VideoWriter(self.out_video, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, dimension)

        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        tracking_object = {}
        prev_cxcy = []
        frame_num = 0

        while True:
            ret, frame = cap.read()
            frame_num += 1

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Predict on image
            detect_params = self.model.predict(source=[frame], conf=0.45, save=True)
            for pred in detect_params:
                cls = self.tensor_to_list(pred.boxes.cls)
                bbox = self.tensor_to_list(pred.boxes.xyxy)
                conf = self.tensor_to_list(pred.boxes.conf)

            # Track detected object current & previous positions
            curr_cxcy = []
            for xy in bbox:
                cx = int((xy[0] + xy[2]) / 2)
                cy = int((xy[1] + xy[3]) / 2)
                center_pt = (cx, cy)
                curr_cxcy.append(center_pt)

            for pt1, pt2 in zip(curr_cxcy, prev_cxcy):
                distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
                if distance < 10:
                    for idx, pt1 in enumerate(curr_cxcy):
                        if (idx == 0) or (idx in track_obID):
                            obID = idx
                            tracking_object[obID] = pt1
                        else:
                            obID = max(track_obID) + 1
                            tracking_object[obID] = pt1

                        cv2.circle(img=frame, center=pt1, radius=1, color=(255, 0, 0), thickness=4)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img=frame, text=str(obID) + " " + str(pt1), org=(pt1[0], pt1[1] - 7),
                                    fontFace=font, fontScale=0.5, color=(255, 255, 255), thickness=1)

            # Draw Kalman prediction
            kalman_xy = self.kalman_pred_coordinate(frame_num)
            for xy in kalman_xy:
                x = int(xy[0])
                y = int(xy[1])
                cv2.circle(img=frame, center=(x, y), radius=1, color=(204, 0, 204), thickness=4)

            track_obID = list(tracking_object.keys())
            prev_cxcy = curr_cxcy.copy()

            # Draw bounding boxes, class, confidence
            if len(cls) != 0:
                for i in range(len(detect_params[0])):
                    clsID = cls[i]
                    bb = bbox[i]
                    cf = conf[i]
                    cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                                  self.detection_colors[int(clsID)], 3)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img=frame, text=self.class_list[int(clsID)] + " " + str(round(cf, 3)),
                                org=(int(bb[0]), int(bb[1]) - 10), fontFace=font, fontScale=0.5, color=(255, 255, 255),
                                thickness=1)

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

# Constants
KALMAN_CSV= 'assignment-3-drone-kalman-filter/3-Kalman/kalman_vid1.csv'
CLASS_DIR = 'assignment-3-drone-kalman-filter/3-Kalman/VisDrone.txt'
MODEL_DIR = 'assignment-3-drone-kalman-filter/2-ODmodel/result/100_epochs/weights/best.pt'
INPUT_VID = 'assignment-3-drone-kalman-filter/DSVIDEOS/1-car.mp4'
OUT_VIDEO = 'assignment-3-drone-kalman-filter/KMVIDEOS/VID1-5-kalman.avi'

# Create object tracker instance and run tracking
tracker = ObjectTracker(KALMAN_CSV, CLASS_DIR, MODEL_DIR, INPUT_VID, OUT_VIDEO)
tracker.track_objects()
