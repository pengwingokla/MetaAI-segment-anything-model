import os
from extract.download_extract_frames import VideoProcessor
from object_detection.main_postion import DetectedPositionCompiler
from kalman.filterpy import KalmanFilter
from kalman.main_kalman import Kalman
from object_detection.main_object_detection import ObjectDetector

def main():
    ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
    # -------------------------------- TASK 1
    # URL_LIST = [
    #     'https://www.youtube.com/watch?v=5dRramZVu2Q&ab_channel=R2bEEaton',
    #     'https://www.youtube.com/watch?v=2NFwY15tRtA&ab_channel=PantelisMonogioudis',
    #     'https://www.youtube.com/watch?v=WeF4wpw7w9k&ab_channel=PantelisMonogioudis',
    #     'https://www.youtube.com/watch?v=2hQx48U1L-Y&ab_channel=victorskrabe'
    # ]
    
    # processor = VideoProcessor(ROOT_DIR, URL_LIST)  # Step 1: Download videos
    # processor.download_video()                      
    # if not os.path.exists(ROOT_DIR):                # Step 2: Extract frames
    #     os.makedirs(ROOT_DIR)
    # processor.process_all_videos()
    # -------------------------------- TASK 2
    
    # See finetuning the model in main_train.ipynb

    # -------------------------------- TASK 2 & TASK 3
    ROOT_VIDEO = ROOT_DIR + '/DSVIDEOS'
    CLASS_DIR = ROOT_DIR +  '/kalman/VisDrone.txt'
    MODEL_DIR = ROOT_DIR + '/object_detection/result/100_epochs/weights/best.pt'

    # Define the output folders for videos and CSVs
    OBJDETECT_VID = 'assignment-3-drone-kalman-filter/OBJDETECT_VID'
    OBJDETECT_CSV = 'assignment-3-drone-kalman-filter/OBJDETECT_CSV'
    KALMAN_VID = 'assignment-3-drone-kalman-filter/KALMAN_VID'
    KALMAN_CSV = 'assignment-3-drone-kalman-filter/KALMAN_CSV'
    os.makedirs(OBJDETECT_VID, exist_ok=True) # if not exist, create
    os.makedirs(OBJDETECT_CSV, exist_ok=True)
    os.makedirs(KALMAN_VID, exist_ok=True)
    os.makedirs(KALMAN_CSV, exist_ok=True)

    # Get a list of all video files in the root video folder
    video_files = [file for file in os.listdir(ROOT_VIDEO) if file.endswith('.mp4')]

    # Iterate over each video file
    video_id = 1
    for video_file in video_files:
        input_vid = os.path.join(ROOT_VIDEO, video_file)
        objdet_vid = os.path.join(OBJDETECT_VID, 'detected_vid%03d.avi' % (video_id))
        objdet_csv = os.path.join(OBJDETECT_CSV, 'detected_vid%03d.csv' % (video_id))

        # Use pretrained object detection model to compile detected object positions
        detector = DetectedPositionCompiler(
            csv_path =objdet_csv,
            class_dir=CLASS_DIR,
            model_dir=MODEL_DIR,
            input_vid=input_vid,
            out_video=objdet_vid
        )
        detector.detect_objects()

        # Use detected center points to compute Kalman
        FNAME_OUT = 'kalman-distribution.png'
        kalman_csv= os.path.join(KALMAN_CSV, 'kalman_vid%03d.csv' % (video_id))

        # kf = KalmanFilter(input_csv=objdet_csv,
        #                   fname_out=FNAME_OUT,
        #                   kalman_csv=kalman_csv)
        # kf.fit_gaussians()
        # kf.plot_histograms()
        # kf.kalman_filter()

        kalman_filter = Kalman()
        kalman_filter.run_kalman_filter(objdet_csv, kalman_csv)

        # Draw trajectories based on Kalman prediction and update
        FPS = 10
        out_video = os.path.join(KALMAN_VID, 'kalman_vid%03d.vid' % (video_id))
        print(kalman_csv)
        object_detection = ObjectDetector(kalman_csv=kalman_csv, 
                                          class_dir=CLASS_DIR, 
                                          model_dir=MODEL_DIR, 
                                          input_vid=input_vid, 
                                          out_video=out_video, 
                                          fps=FPS)
        object_detection.run_detection()

        video_id += 1

if __name__ == "__main__":
    main()