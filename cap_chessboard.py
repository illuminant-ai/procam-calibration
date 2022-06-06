"""
Capture projection pattern and decode x-coordinate.
"""
import os
import glob

import cv2
import numpy as np

import fullscreen.fullscreen as fs

CAMERA_RESOLUTION = (1920, 1080)
VIDEO_BUFFER_LEN = 5                    ## Changed to reflect flush_cap(cap).

def flush_cap(cap):
    """
    Attempts to flush the video capture buffer. This is needed because
    the first VIDEO_BUFFER_LEN frames of a series of captures essentially
    won't update properly / will be stale. This is unfortunately a hardware
    issue and can't be easily disabled.
    """
    for _ in range(VIDEO_BUFFER_LEN):   ## Use VIDEO_BUFFER_LEN.
        cap.grab()


def imshowAndCapture(cap, pattern, screen, delay=500):
    """
    Projects a graycode pattern, captures the scene from the camera after a
    specified amount of delay, and returns a grayscale version of the
    captured image.
    """
    screen.imshow(pattern)
    cv2.waitKey(delay)
    flush_cap(cap)
    _, img_frame = cap.read()           ## Renamed for readability
    img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    return img_gray


def main():
    # Set up the camera for capture.
    """
    IMPORTANT:

    Change this value to reflect the actual index of the camera.
    If your computer already has a webcam, the index of the externally linked
    camera may be 1 instead of 0.
    """
    cap = cv2.VideoCapture(1)       # Index of the camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
   
    # Read the graycode pattern files.
    patterns = [None] * len(os.listdir('./graycode_pattern/'))      ## Renamed for readability.
    for filename in os.listdir('./graycode_pattern/'):
        if filename.endswith(".png"): 
            image = cv2.imread('./graycode_pattern/'+filename)
            # Extract the index from filename "pattern_<index>.png"
            position = int(filename[8:10])
            patterns[position] = image
        else:
            continue

    """
    for i in range(VIDEO_BUFFER_LEN):
        patterns.append(patterns[-1])
    """

    # Set up the projector screen to have a blank display.
    screen = fs.FullScreen('cv2', screen_id=1)
    black = np.zeros(CAMERA_RESOLUTION)
    screen.imshow(black)

    # Capture a sequence of scenes with different graycode patterns projected.
    ## flush_cap(cap)      ## Removed because imshowAndCapture flushes before capturing
    imlist = [imshowAndCapture(cap, pat, screen, 100) for pat in patterns]

    """
    # Chop off lagging frames
    # imlist = imlist[VIDEO_BUFFER_LEN:]
    """

    # Create a new capture directory and save the graycode-pattern-projected
    # images into this directory. The directory is structured as follows:
    #   ./capture_<n>/ --- graycode_00.png
    #                   |- graycode_01.png
    #                   |       .
    #                   |       .
    #                   |- graycode_<m>.png    
    dirnames = sorted(glob.glob('./capture_*'), key = lambda x: int(x.split('_')[1]))
    if len(dirnames) == 0:
        most_recent_capture = "./capture_-1"
    else:
        most_recent_capture = dirnames[-1]
    tokenized = most_recent_capture.split('_')
    most_recent_index = int(tokenized[1])
    new_capture_index = most_recent_index + 1
    new_capture_dir = tokenized[0] + '_' + str(new_capture_index) + '/'
    os.mkdir(new_capture_dir)
    print("Saving to " + new_capture_dir)
    for index, img in enumerate(imlist):
        if(index < 10):
            str_ind = '0' + str(index)
        else:
            str_ind = str(index)
        cv2.imwrite(new_capture_dir + "graycode_" + str_ind + ".png", img)
        ## Suggestion: cv2.imwrite(new_capture_dir + "graycode_" + str(index).zfill(2) + ".png", img)

    # Close the projector and the camera.
    cv2.destroyAllWindows()     ## Why not: screen.destroyWindow() as there is only a single GUI.
    cap.release()


if __name__=="__main__":
    main()
