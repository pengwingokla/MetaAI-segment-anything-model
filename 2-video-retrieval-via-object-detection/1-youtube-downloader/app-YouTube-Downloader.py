import customtkinter as ctk
from tkinter import ttk
from  pytube import YouTube
import os
import re
from youtube_transcript_api import YouTubeTranscriptApi

def download_video():
    url = url_entry.get()
    resolution = resolution_var.get()

    progress_label.pack(pady=10)
    vid_status.pack(pady=10)
    sub_status.pack(pady=10)

    try:
        yt = YouTube(url, on_progress_callback=on_progress)
        stream = yt.streams.filter(res=resolution).first()
        yt_name= format_video_title(yt.title)
        print(yt_name)

        # download the video to a specified directory
        stream.download(output_path="video-downloads",filename=yt_name+'.mp4')
        
        # update video download status
        vid_status.configure(text="Video Downloaded", text_color="green")
        
        # download the subtitles
        auto_caption = yt.captions['a.en']
        sub_filename = os.path.join("subtitle-downloads", f"{yt_name}.xml")
        with open(sub_filename, "w", encoding="utf-8") as file:
            file.write(auto_caption.xml_captions)
        
        # update subtitle download status
        sub_status.configure(text="Subtitles Downloaded", text_color="green")
        
    except Exception as e:
        vid_status.configure(text=f"Error {str(e)}", text_color="white", fg_color="red")
        sub_status.configure(text=f"Error {str(e)}", text_color="white", fg_color="red")
                                                                                                                                                                                                   
def on_progress(stream, chunk, bytes_remaining):
    total_size = stream.filesize
    bytes_downloaded = total_size - bytes_remaining
    percentage = bytes_downloaded / total_size * 100

    progress_label.configure(text=str(int(percentage)) + "%")
    progress_label.update()
    progress_bar.set(float(percentage/100))

def format_video_title(title):
    # Get the substring before "| NPR"
    title = title.split('| NPR')[0].strip()
    # Convert to lowercase
    title = title.lower()
    # Remove punctuation and replace with dashes
    title = re.sub(r'[^\w\s]', '', title)
    title = re.sub(r'\s+', '-', title)
    return title

# create a root window
root = ctk.CTk()
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# title of the window
root.title("YouTube Downloader")

# set min and max width and the height
root.geometry("720x480")
root.minsize(480, 480)
root.maxsize(1080, 720)

# frame to hold the content
content_frame = ctk.CTkFrame(root)
content_frame.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)
progress_frame = ctk.CTkFrame(root)
progress_frame.pack(fill=ctk.BOTH, side="top", padx=10, pady=10)

# define fonts
heading = ctk.CTkFont(family="San Francisco", size=20)
body = ctk.CTkFont(family="San Francisco", size=10)

# label and the entry widget for the video url
url_label = ctk.CTkLabel(content_frame, text="YouTube Downloader", font=heading)
url_label.pack(pady=10)
url_entry = ctk.CTkEntry(content_frame, placeholder_text="Enter url here", font=body, width=400, height=40)
url_entry.pack(pady=10)

# create a resolutions combo box
resolutions = ["720p","360p","240p"]
resolution_var = ctk.StringVar()
resolution_combobox = ttk.Combobox(content_frame, values=resolutions, textvariable=resolution_var)
resolution_combobox.pack(pady=10)
resolution_combobox.set("720p")

# create a download button
download_button = ctk.CTkButton(content_frame, text="Download", font=body, command=download_video)
download_button.pack(pady=10)

# create a label and the progress bar to display the download progress
progress_label = ctk.CTkLabel(progress_frame, text="0%", font=body)
progress_bar = ctk.CTkProgressBar(progress_frame, width=400)
progress_bar.set(0)
progress_label.pack(padx=10, pady=10, side="right")
progress_bar.pack(padx=10, pady=10, side="right")

# create a status label
vid_status = ctk.CTkLabel(content_frame, text="Video Downloaded", font=body)
sub_status = ctk.CTkLabel(content_frame, text="Subtitles Downloaded", font=body)

# start the app
root.mainloop()