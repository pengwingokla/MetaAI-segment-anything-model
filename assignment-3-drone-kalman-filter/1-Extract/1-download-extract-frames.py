from pytube import YouTube
import os
import re
import cv2


def format_video_title(title):
    # Get the substring before "| NPR"
    title = title.split('| NPR')[0].strip()
    # Convert to lowercase
    title = title.lower()
    # Remove punctuation and replace with dashes
    title = re.sub(r'[^\w\s]', '', title)
    title = re.sub(r'\s+', '-', title)
    return title

def download_video(url_list, output_dir):
    for id, url in enumerate(url_list, 1):
        yt = YouTube(url)
        stream = yt.streams.filter(res='1080').first()
        yt_name= format_video_title(yt.title)

        # download the video to a specified directory
        stream.download(output_path=output_dir,filename=f'{id}-'+yt_name+'.mp4')

def extract_frame(video_file, output_directory, video_id):
    print("PROCESSING VIDEO: ", video_file)
    video = cv2.VideoCapture(video_file)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    TIME_INTERVAL = 1 #seconds                              <------- CHANGE HERE
    frame_count = 0

    while(True):
        ret, frame = video.read()
        if not ret: 
            break
    
        if frame_count % (fps * TIME_INTERVAL) == 0:
            output_path = os.path.join(output_directory, '%03d_%d.jpg' % (video_id, frame_count))

            cv2.imwrite(output_path, frame)
        
            # Calculate progress percentage
            progress_percent = (frame_count / total_frames) * 100
            print("Extracting frames..........%.2f%%" % progress_percent)

        frame_count += 1
  
    video.release() 
    cv2.destroyAllWindows()

def process_all_videos(vid_input_dir: str, root_output_dir: str):
    """This function extract frame from all video in the specified directory"""
    video_id = 1
    for filename in os.listdir(vid_input_dir):
        if filename.endswith(".mp4"):
            video_path = os.path.join(vid_input_dir, filename)
            output_dir = root_output_dir + "/FRAMES-" + os.path.splitext(filename)[0]
            os.makedirs(output_dir, exist_ok=True)
            extract_frame(video_path, output_dir, video_id)
            # extract_timestamp(video_path, timestamp_dir, video_id)
            video_id += 1

# Main function
def main():
    # Step 1: Download videos
    video_dir = "DSVIDEO-drone"
    url_list = ['https://www.youtube.com/watch?v=5dRramZVu2Q&ab_channel=R2bEEaton',
             'https://www.youtube.com/watch?v=2NFwY15tRtA&ab_channel=PantelisMonogioudis',
             'https://www.youtube.com/watch?v=WeF4wpw7w9k&ab_channel=PantelisMonogioudis']

    # download_video(url_list, output_dir=video_dir)
    
    # Step 2: Extract frames
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    rootframe_dir = "DSFRAMES"
    process_all_videos(video_dir, rootframe_dir)

if __name__ == "__main__":
    main()