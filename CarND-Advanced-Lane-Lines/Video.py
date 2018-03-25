#
import numpy as np 
import pandas as pd  
import os

from moviepy.editor import VideoFileClip
#from IPython.display import HTML
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


from Parameter import ParametersClass

from LaneDetector_Challenge2 import LaneDetector

def main():

    paramCls = ParametersClass()
    params = paramCls.initialize()
 
    
    print("-"*30)
    print("* incoming video file name *",params.infile)
    print("-"*30)
 
    input_video = params.infile

    output_video = os.path.join("output_images",input_video )

    lane_detector = LaneDetector()

    #lane_detector.image_pipeline(test_image)


    clip1 = VideoFileClip(input_video)
    video_clip = clip1.fl_image(lane_detector.image_pipeline)
    #video_clip = clip1.fl_image(lane_detector.test)
    video_clip.write_videofile(output_video, audio=False)



if __name__ == "__main__":
    main()