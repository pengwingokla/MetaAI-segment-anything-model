import cv2
import os 

def extract_frame(video_file, output_directory, video_id):
    print("PROCESSING VIDEO: ", video_file)
    video = cv2.VideoCapture(video_file)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    time_interval = 5 #seconds
    frame_count = 0

    while(True):
        ret, frame = video.read()
        if not ret: 
            break
    
        if frame_count % (fps * time_interval) == 0:
            output_path = os.path.join(output_directory, '%03d_%d.jpg' % (video_id, frame_count))

            cv2.imwrite(output_path, frame)
        
            # Calculate progress percentage
            progress_percent = (frame_count / total_frames) * 100
            print("Extracting frames..........%.2f%%" % progress_percent)

        frame_count += 1
  
    video.release() 
    cv2.destroyAllWindows()

def extract_timestamp(video_file, output_directory, video_id):
    video = cv2.VideoCapture(video_file)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    time_interval = 5 #seconds
    frame_count = 0

    while(True):
        ret, _ = video.read()
        if not ret: 
            break
    
        if frame_count % (fps * time_interval) == 0:
            # Calculate timestamp
            current_timestamp = video.get(cv2.CAP_PROP_POS_MSEC)
            minutes = int((current_timestamp / 1000) / 60)
            seconds = int((current_timestamp / 1000) % 60)
            formatted_timestamp = f'{minutes:02d}:{seconds:02d}'
            
            # Store timestamp in a text file
            with open(os.path.join(output_directory, '%03d_%d.txt' % (video_id, frame_count)), 'w') as f:
                f.write(formatted_timestamp)
        
        frame_count += 1
  
    video.release()

def process_all_videos(vid_input_dir: str, root_output_dir: str, timestamp_dir:str):
    """This function extract frame from all video in the specified directory"""
    video_id = 1
    for filename in os.listdir(vid_input_dir):
        if filename.endswith(".mp4"):
            video_path = os.path.join(vid_input_dir, filename)
            output_dir = root_output_dir + "/DATASET-FRAMES-" + os.path.splitext(filename)[0]
            os.makedirs(output_dir, exist_ok=True)
            extract_frame(video_path, output_dir, video_id)
            extract_timestamp(video_path, timestamp_dir, video_id)
            video_id += 1

# Main function
def main():
    
    if not os.path.exists("DATASET-frames"):
        os.makedirs("DATASET-frames")

    video_directory = "assignment-2-video-search\\1-youtube-downloader\\video-downloads"
    root_frame_dir = "assignment-2-video-search\\1-youtube-downloader\\DATASET-FRAMES"
    timestamp_dir = "assignment-2-video-search\\1-youtube-downloader\\DATASET-TIMESTAMP"
    process_all_videos(video_directory, root_frame_dir, timestamp_dir)

if __name__ == "__main__":
    main()