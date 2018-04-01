#
from io import BytesIO
import cv2
import numpy as np
from tqdm import tqdm
from IPython.display import Image
import matplotlib as mpl
from moviepy.editor import VideoFileClip



def load_test_video(file_name='test_video.mp4'):
    vimages = []
    vframes = []
    count = 0
    clip = VideoFileClip(file_name)
    for img in clip.iter_frames(progress_bar=True):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        vimages.append(img)
        vframes.append("%s - %d" % (file_name, count))
        count += 1

    return vimages, vframes


def arr2img(arr):
    """Display a 2- or 3-d numpy array as an image."""
    if arr.ndim == 2:
        format, cmap = 'png', mpl.cm.gray
    elif arr.ndim == 3:
        format, cmap = 'jpg', None
    else:
        raise ValueError("Only 2- or 3-d arrays can be displayed as images.")
    # Don't let matplotlib autoscale the color range so we can control
    # overall luminosity
    vmax = 255 if arr.dtype == 'uint8' else 1.0
#     vmax=1.0
    with BytesIO() as buffer:
        mpl.image.imsave(buffer, arr, format=format, cmap=cmap,
                         vmin=0, vmax=vmax)
        out = buffer.getvalue()
    return Image(out)
