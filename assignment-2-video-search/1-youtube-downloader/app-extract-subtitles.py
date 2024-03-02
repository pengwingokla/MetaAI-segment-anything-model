from youtube_transcript_api import YouTubeTranscriptApi
import os

root_folder = "assignment-2-video-search\\1-youtube-downloader\\subtitle-downloads"
vid = ['wbWRWeVe1XE&ab_channel=NPR', 
       'FlJoBhLnqko&ab_channel=NPR', 
       'Y-bVwPRy_no&ab_channel=NPR']

for video in vid:
    srt = YouTubeTranscriptApi.get_transcript(video)
    text = ""
    subID = 1

    filename = os.path.join(root_folder, f"\\SUB00{subID}.txt")
    with open(filename, "w") as file:
        for i in srt:
            text += i["text"]
        file.write(text)
    os.startfile(filename)