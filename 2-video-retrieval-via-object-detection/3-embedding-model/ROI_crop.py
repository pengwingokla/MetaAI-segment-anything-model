from PIL import Image
import pandas as pd

def crop_roi(image_path, x_topleft, y_topleft, x_bottomright, y_bottomright):
    image = Image.open(image_path)
    cropped_image = image.crop((x_topleft, y_topleft, x_bottomright, y_bottomright))
    return cropped_image

roi_csv = "assignment-2-video-search\\3-embedding-model\\object_data.csv"
roi_dir = "assignment-2-video-search\\3-embedding-model\\image-cropped-rois"

df = pd.read_csv(roi_csv, dtype={'videoID': str})
crop_id_counter = 1
# Iterate over rows in the DataFrame
for index, row in df.iterrows():
    video_id = row['videoID']
    frame_id = row['frameID']
    x_topleft = row['x_topleft']
    y_topleft = row['y_topleft']
    x_bottomright = row['x_bottomright']
    y_bottomright = row['y_bottomright']
    
    # Generate image path based on your file naming convention
    image_path = f"assignment-2-video-search\\1-youtube-downloader\\DATASET-FRAMES\\{video_id}\\{frame_id}.jpg"
    
    # Crop ROI
    cropped_image = crop_roi(image_path, x_topleft, y_topleft, x_bottomright, y_bottomright)
    
    # Save cropped image
    crop_id = f"{crop_id_counter:03d}"
    cropped_image.save(f"assignment-2-video-search/3-embedding-model/image-cropped-rois/{video_id}_{frame_id}_CROP{crop_id}.jpg")
    crop_id_counter += 1 



