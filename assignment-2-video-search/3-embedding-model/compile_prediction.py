from ultralytics import YOLO
from matplotlib import pyplot as plt
import torch
import csv
import os

def draw_bbox(image_path, model_path, save_path):
    """Detect object, lot bbox, and store"""
    image_filename = os.path.basename(image_path)[:-4]
    model = YOLO(model_path)
    pred_imgs = model.predict(image_path, conf=0.3)
        
    # Iterate over each prediction, plot, and save
    for i, pred_img in enumerate(pred_imgs):
        result_array = pred_img.plot()
        plt.imshow(result_array)
        plt.title(f"pred_{image_filename}") 
        plt.savefig(f'{save_path}\\pred_{image_filename}.jpg', dpi=300)

def get_bbox_info(model_path, image_path):
    """Obtain bounding box attributes"""
    model = YOLO(model_path)
    if image_path.endswith('.jpg'):        
        pred = model(image_path)
        for result in pred:
            boxes = result.boxes
            label= boxes.cls  # class values of the boxes
            conf = boxes.conf # confidence values of the boxes
            xyxy = boxes.xyxy # boxes top left and bottom right
            xywh = boxes.xywh # top left width height
        return label, conf, xyxy, xywh

def tensor_to_list(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.tolist()
    else:
        raise TypeError("Input must be a PyTorch tensor")
    
def generate_csv(video_id, frame_id, label, conf, coordinates, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['videoID', 'frameID', 'label', 'confidence', 
                      'x_topleft', 'y_topleft', 'width', 'height']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()

        for label, conf, xyxy in zip(label, conf, coordinates):
            x_topleft, y_topleft, width, height = xyxy
            writer.writerow({'videoID': video_id,'frameID': frame_id, 
                             'label': label, 'confidence': conf,
                             'x_topleft': x_topleft, 'y_topleft': y_topleft,
                             'width': width, 'height': height})
            
def main():
    model_path  = 'assignment-2-video-search\\3-embedding-model\\best.pt'
    # root_DATASET = 'assignment-2-video-search\\1-youtube-downloader\\DATASET-FRAMES'
    # save_path   = 'assignment-2-video-search\\3-embedding-model\\image-pred-bbox'

    root_DATASET = 'assignment-2-video-search\\3-embedding-model\\image-pred'
    save_path = 'assignment-2-video-search\\3-embedding-model\\image-pred-bbox'
    
    video_id_counter = 1
    for folder_name in sorted(os.listdir(root_DATASET)):
        folder_DATASET = os.path.join(root_DATASET, folder_name)
        video_id = f"{video_id_counter:03d}" 
        video_id_counter += 1 
        for image in sorted(os.listdir(folder_DATASET)):
            image_path = os.path.join(folder_DATASET, image)
            image_id   = os.path.splitext(image)[0]
            
            # Use model for object detection, draw bbox and store
            # draw_bbox(image_path, model_path, save_path)

            # Get bbox info and compile results in csv
            bbox_info  = get_bbox_info(model_path, image_path)
            label = tensor_to_list(bbox_info[0])
            conf  = tensor_to_list(bbox_info[1])
            xyxy  = tensor_to_list(bbox_info[2])
            if len(label) == 0:
                continue
            # Use model for object detection, draw bbox and store
            draw_bbox(image_path, model_path, save_path)
            generate_csv(video_id= video_id,
                         frame_id= image_id,
                         label   = label,
                         conf    = conf,
                         coordinates  = xyxy,
                         csv_filename='object_data.csv')
            

if __name__ == "__main__":
    main()
    
