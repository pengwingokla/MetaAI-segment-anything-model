from pytube import YouTube
import os
import re
import cv2

class VideoProcessor:
    def __init__(self, root_dir, url_list):
        self.root_dir = root_dir
        self.video_dir = os.path.join(root_dir, "DSVIDEOS")
        self.frame_dir = os.path.join(root_dir, "DSFRAMES")
        self.url_list = url_list

    @staticmethod
    def format_video_title(title):
        # Get the substring before "| NPR"
        title = title.split('| NPR')[0].strip()
        # Convert to lowercase
        title = title.lower()
        # Remove punctuation and replace with dashes
        title = re.sub(r'[^\w\s]', '', title)
        title = re.sub(r'\s+', '-', title)
        return title

    def download_video(self):
        for id, url in enumerate(self.url_list, 1):
            yt = YouTube(url)
            stream = yt.streams.filter(res='720p').first()
            yt_name = self.format_video_title(yt.title)

            # download the video to a specified directory
            print('Downloading...', yt_name)
            stream.download(output_path=self.video_dir, filename=f'{id}-'+'car.mp4')

    @staticmethod
    def extract_frame(video_file, output_directory, video_id):
        print("PROCESSING VIDEO: ", video_file)
        video = cv2.VideoCapture(video_file)
        fps = int(video.get(cv2.CAP_PROP_FPS))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        TIME_INTERVAL = 1  # seconds                              <------- CHANGE HERE
        frame_count = 0

        while True:
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

    def process_all_videos(self):
        """This function extract frame from all video in the specified directory"""
        video_id = 1
        for filename in os.listdir(self.video_dir):
            if filename.endswith(".mp4"):
                video_path = os.path.join(self.video_dir, filename)
                output_dir = self.frame_dir + '\FRAME' + os.path.splitext(filename)[0]
                os.makedirs(output_dir, exist_ok=True)
                self.extract_frame(video_path, output_dir, video_id)
                video_id += 1

