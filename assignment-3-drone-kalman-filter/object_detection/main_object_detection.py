import math
import random
import torch
import cv2
import csv
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, kalman_csv, class_dir, model_dir, input_vid, out_video, fps=10):
        self.KALMAN_CSV = kalman_csv
        self.CLASS_DIR = class_dir
        self.MODEL_DIR = model_dir
        self.INPUT_VID = input_vid
        self.OUT_VIDEO = out_video
        self.FPS = fps

        self.model = YOLO(self.MODEL_DIR)
        self.object_cls = open(self.CLASS_DIR, "r").read().split("\n")
        self.detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(self.object_cls))]
        self.cap = cv2.VideoCapture(self.INPUT_VID)
        self.frame_wid = int(self.cap.get(3))
        self.frame_hyt = int(self.cap.get(4))
        self.dimension = (self.frame_wid, self.frame_hyt)
        self.out = cv2.VideoWriter(self.OUT_VIDEO, cv2.VideoWriter_fourcc(*'MJPG'), self.FPS, self.dimension)

    def tensor_to_list(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.tolist()
        else:
            raise TypeError("Input must be a PyTorch tensor")

    def kalman_pred_coordinate(self, frame_num):
        THRES_FRAME = 10
        coordinates = []
        with open(self.KALMAN_CSV, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Skip header row
            for row in csvreader:
                if int(row[0]) <= frame_num and int(row[0]) >= frame_num - THRES_FRAME:
                    coordinates.append((float(row[1]), float(row[2])))
        return coordinates

    def detect_objects(self, frame):
        detect_params = self.model.predict(source=[frame], conf=0.45, save=True)
        bbox = []
        cls = []
        conf = []
        for pred in detect_params:
            cls += self.tensor_to_list(pred.boxes.cls)
            bbox += self.tensor_to_list(pred.boxes.xyxy)
            conf += self.tensor_to_list(pred.boxes.conf)
        return cls, bbox, conf

    def run_detection(self):
        tracking_object = {}
        prev_cxcy = []
        frame_num = 0

        while True:
            ret, frame = self.cap.read()
            frame_num += 1

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            cls, bbox, conf = self.detect_objects(frame)

            curr_cxcy = []
            for xy in bbox:
                cx = int((xy[0] + xy[2]) / 2)
                cy = int((xy[1] + xy[3]) / 2)
                center_pt = (cx, cy)
                curr_cxcy.append(center_pt)

            for pt1, pt2 in zip(curr_cxcy, prev_cxcy):
                distance = math.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1])
                if distance < 10:
                    for idx, pt1 in enumerate(curr_cxcy):
                        if (idx == 0) or (idx in track_obID):
                            obID = idx
                            tracking_object[obID] = pt1
                        else:
                            obID = max(track_obID) + 1
                            tracking_object[obID] = pt1

                        cv2.circle(img=frame, center=pt1, radius=1, color=(255,0,0), thickness=4)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img=frame, text=str(obID)+" "+str(pt1), org=(pt1[0],pt1[1]-7), fontFace=font, fontScale=0.5, color=(255, 255, 255), thickness=1,)

            kalman_xy = self.kalman_pred_coordinate(frame_num)
            for xy in kalman_xy:
                x = int(xy[0]) ; y = int(xy[1])
                cv2.circle(img=frame, center=(x,y), radius=1, color=(204,0,204), thickness=4)

            track_obID = list(tracking_object.keys())
            prev_cxcy = curr_cxcy.copy()

            if len(cls) != 0:
                for i in range(len(cls)):
                    clsID = cls[i]
                    bb = bbox[i]
                    cf = conf[i]
                    cv2.rectangle(frame,(int(bb[0]), int(bb[1])),(int(bb[2]), int(bb[3])), self.detection_colors[int(clsID)],3)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img=frame,text=self.object_cls[int(clsID)] + " " + str(round(cf, 3)),org=(int(bb[0]), int(bb[1]) - 10),fontFace=font,fontScale=0.5,color=(255, 255, 255),thickness=1)

            self.out.write(frame)
            cv2.imshow("ObjectDetection", frame)

            if cv2.waitKey(1) == ord("q"):
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
