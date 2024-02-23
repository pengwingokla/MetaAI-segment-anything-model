import cv2
import os 

def extract_frame(video_file, output_directory):
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
            output_path = os.path.join(output_directory, '%d.jpg' % frame_count)
            cv2.imwrite(output_path, frame)
        
            # Calculate progress percentage
            progress_percent = (frame_count / total_frames) * 100
            print("Extracting frames..........%.2f%%" % progress_percent)

        frame_count += 1
  
    video.release() 
    cv2.destroyAllWindows()


def process_all_videos(vid_input_dir: str, root_output_dir: str):
    """This function extract frame from all video in the specified directory"""
    for filename in os.listdir(vid_input_dir):
        if filename.endswith(".mp4"):
            video_path = os.path.join(vid_input_dir, filename)
            output_dir = root_output_dir + "/DATASET-FRAMES-" + os.path.splitext(filename)[0]
            os.makedirs(output_dir, exist_ok=True)
            extract_frame(video_path, output_dir)


# Main function
def main():
    
    if not os.path.exists("DATASET-frames"):
        os.makedirs("DATASET-frames")

    video_directory = "video-downloads"
    output_root_dir = "DATASET-FRAMES"

    process_all_videos(video_directory, output_root_dir)

if __name__ == "__main__":
    main()