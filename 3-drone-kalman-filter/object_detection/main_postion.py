import csv
import math
import random
import torch
import cv2
import numpy as np
from ultralytics import YOLO

class DetectedPositionCompiler:
    def __init__(self, csv_path, class_dir, model_dir, input_vid, out_video, fps=10):
        self.CSV_PATH = csv_path
        self.CLASS_DIR = class_dir
        self.MODEL_DIR = model_dir
        self.INPUT_VID = input_vid
        self.OUT_VIDEO = out_video
        self.FPS = fps
        self.detect_colors = []
        self.tracking_object = {}
        self.prev_cxcy = []
        self.frame_num = 0

        self._load_class_list()
        self._generate_detection_colors()
        self._initialize_video_capture()
        self._initialize_output_video_writer()
        self._initialize_model()

    def _load_class_list(self):
        with open(self.CLASS_DIR, "r") as my_file:
            data = my_file.read()
            self.class_list = data.split("\n")

    def _generate_detection_colors(self):
        for _ in range(len(self.class_list)):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            self.detect_colors.append((b, g, r))

    def _initialize_video_capture(self):
        self.cap = cv2.VideoCapture(self.INPUT_VID)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

    def _initialize_output_video_writer(self):
        frame_wid = int(self.cap.get(3))
        frame_hyt = int(self.cap.get(4))
        dimension = (frame_wid, frame_hyt)
        self.out = cv2.VideoWriter(self.OUT_VIDEO, cv2.VideoWriter_fourcc(*'MJPG'), self.FPS, dimension)

    def _initialize_model(self):
        self.model = YOLO(self.MODEL_DIR)

    def detect_objects(self):
        with open(self.CSV_PATH, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Frame', 'Object ID', 'Center X', 'Center Y'])

            while True:
                ret, frame = self.cap.read()
                self.frame_num += 1

                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break

                detect_params = self.model.predict(source=[frame], conf=0.45, save=True)
                for pred in detect_params:
                    cls = tensor_to_list(pred.boxes.cls)
                    bbox = tensor_to_list(pred.boxes.xyxy)
                    conf = tensor_to_list(pred.boxes.conf)

                curr_cxcy = []
                for xy in bbox:
                    cx = int((xy[0] + xy[2]) / 2)
                    cy = int((xy[1] + xy[3]) / 2)
                    center_pt = (cx, cy)
                    curr_cxcy.append(center_pt)

                for pt1, pt2 in zip(curr_cxcy, self.prev_cxcy):
                    distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
                    if distance < 10:
                        for idx, pt1 in enumerate(curr_cxcy):
                            if (idx == 0) or (idx in self.track_obID):
                                obID = idx
                                self.tracking_object[obID] = pt1
                                csvwriter.writerow([int(self.frame_num), obID, pt1[0], pt1[1]])
                            else:
                                obID = max(self.track_obID) + 1
                                self.tracking_object[obID] = pt1
                                csvwriter.writerow([int(self.frame_num), obID, pt1[0], pt1[1]])
                            cv2.circle(img=frame, center=pt1, radius=1, color=(255, 0, 0), thickness=4)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(img=frame, text=str(obID) + " " + str(pt1), org=(pt1[0], pt1[1] - 7),
                                        fontFace=font, fontScale=0.5, color=(255, 255, 255), thickness=1)

                self.track_obID = list(self.tracking_object.keys())
                self.prev_cxcy = curr_cxcy.copy()

                if len(cls) != 0:
                    for i in range(len(detect_params[0])):
                        clsID = cls[i]
                        bb = bbox[i]
                        cf = conf[i]
                        cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                                      self.detect_colors[int(clsID)], 3)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img=frame, text=self.class_list[int(clsID)] + " " + str(round(cf, 3)),
                                    org=(int(bb[0]), int(bb[1]) - 10), fontFace=font, fontScale=0.5,
                                    color=(255, 255, 255), thickness=1)

                self.out.write(frame)
                cv2.imshow("ObjectDetection", frame)
                if cv2.waitKey(1) == ord("q"):
                    break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

def tensor_to_list(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.tolist()
    else:
        raise TypeError("Input must be a PyTorch tensor")
