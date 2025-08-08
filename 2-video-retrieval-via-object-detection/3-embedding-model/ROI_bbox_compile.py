from ultralytics import YOLO
from matplotlib import pyplot as plt
import torch
import csv
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
            label= tensor_to_list(boxes.cls)  # class values of the boxes
            conf = tensor_to_list(boxes.conf) # confidence values of the boxes
            xyxy = tensor_to_list(boxes.xyxy) # boxes top left and bottom right
            xywh = tensor_to_list(boxes.xywh) # top left width height
    
        return label, conf, xyxy, xywh

def tensor_to_list(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.tolist()
    else:
        raise TypeError("Input must be a PyTorch tensor")
    
def generate_csv(video_id, frame_id, label, conf, coordinates, csv_filename):
    mode = 'a' if os.path.exists(csv_filename) else 'w'
    with open(csv_filename, mode, newline='') as csvfile:
        fieldnames = ['videoID', 'frameID', 'label', 'confidence', 
                      'x_topleft', 'y_topleft', 'x_bottomright', 'y_bottomright']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header only if the file is newly created
        if mode == 'w':
            writer.writeheader()

        for label, conf, xyxy in zip(label, conf, coordinates):
            x_topleft, y_topleft, x_bottomright, y_bottomright = xyxy
            writer.writerow({'videoID': video_id,'frameID': frame_id, 
                             'label': label, 'confidence': conf,
                             'x_topleft': x_topleft, 'y_topleft': y_topleft,
                             'x_bottomright': x_bottomright, 'y_bottomright': y_bottomright})
            
def main():
    model_path  = 'assignment-2-video-search\\3-embedding-model\\best.pt'
    root_DATASET = 'assignment-2-video-search\\1-youtube-downloader\\DATASET-FRAMES'
    save_path   = 'assignment-2-video-search\\3-embedding-model\\image-pred-bbox'

    # root_DATASET = 'assignment-2-video-search\\3-embedding-model\\image-pred'
    # save_path = 'assignment-2-video-search\\3-embedding-model\\image-pred-bbox'
    
    video_id_counter = 1
    for folder_name in sorted(os.listdir(root_DATASET)):
        folder_DATASET = os.path.join(root_DATASET, folder_name)
        video_id = f"VID{video_id_counter:03d}" 
        video_id_counter += 1 
        for image in sorted(os.listdir(folder_DATASET)):
            image_path = os.path.join(folder_DATASET, image)
            image_id   = os.path.splitext(image)[0]
            
            # Get bbox info and compile results in csv
            label, conf, xyxy, xywh  = get_bbox_info(model_path, image_path)
            if len(label) == 0:
                continue
            # Use model for object detection, draw bbox and store
            # draw_bbox(image_path, model_path, save_path)
            generate_csv(video_id= video_id,
                         frame_id= image_id,
                         label   = label,
                         conf    = conf,
                         coordinates  = xyxy,
                         csv_filename='ROI_data.csv')
            

if __name__ == "__main__":
    main()
    
