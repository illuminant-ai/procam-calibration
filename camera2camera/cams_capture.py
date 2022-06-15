#!/usr/bin/env python3

import cv2
import glob
import os
import tkinter as tk
from PIL import Image, ImageTk

CAMERA_RESOLUTION = (1920, 1080)
FLUSH_COUNT = 5

def flush(camera, count = FLUSH_COUNT):
    """ Flushes the camera buffer to ready for the next capture. """
    for _ in range(count):
        camera.grab()

def setup_camera(index, label=""):
    """ Fetches the camera with the specified index and sets it up. """
    camera = cv2.VideoCapture(index)
    # Tries to set the capture properties of the camera.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    # First and foremost flush before a camera capture.
    flush(camera, FLUSH_COUNT * 2)
    return camera

def capture_image(camera):
    """ Captures a grayscale image from the camera and returns it. """
    flush(camera)
    _, image = camera.read()
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image_grayscale

def save(depth_image, rgb_image):
    """
    Save the two images -- one from the functional one that can measure
    depth and another from the dysfunctional one that can only measure
    RGB -- into an indexed capture directory.
    """
    def get_next_directory():
        dirs = sorted(glob.glob("./captures/capture_*"), key=lambda x: int(x.split("_")[1]))
        recent = "./captures/capture_-1" if len(dirs) == 0 else dirs[-1]
        recent_index = int(recent.split("_")[1])
        next_index = recent_index + 1
        return f"./captures/capture_{next_index}"

    if not os.path.exists("./captures"):
        os.mkdir("./captures")
    dir = get_next_directory()
    print(f"Saving images to the capture director {os.path.normcase(dir)} ...")
    os.mkdir(dir)
    cv2.imwrite(f"{dir}/depth.png", depth_image)
    cv2.imwrite(f"{dir}/rgb.png", rgb_image)

def stream_camera(view, camera):
    """ Stream the images taken by camera into a tkinter view. """
    cv_image = cv2.cvtColor(camera.read()[1], cv2.COLOR_BGR2RGB)
    downsampled = cv2.resize(cv_image, (480, 270))          # (16, 9) * 30
    normal_image = Image.fromarray(downsampled)
    tk_image = ImageTk.PhotoImage(image=normal_image)
    view.image = tk_image
    view.configure(image=tk_image)
    view.after(50, lambda : stream_camera(view, camera))

def setup_gui(capture_fn, depth_cam, rgb_cam):
    """
    Sets up and runs the graphical user interface for capturing and saving
    images.
    """
    window = tk.Tk()
    window.title("Camera Capturer")

    depth_view = tk.Label(master=window)
    stream_camera(depth_view, depth_cam)
    depth_view.pack()

    rgb_view = tk.Label(master=window)
    stream_camera(rgb_view, rgb_cam)
    rgb_view.pack()

    capture = tk.Button(master=window, text="Capture", command=capture_fn)
    capture.pack(fill=tk.X)

    window.mainloop()

def main():
    # ADJUST these indices to reflect the actual indices of the camera.
    print("Setting up the cameras for capture ...")
    depth_cam = setup_camera(2, "depth")
    rgb_cam = setup_camera(1, "rgb")

    def capture():
        """ Captures an image from the two cameras and saves them. """
        # These does NOT imply a depth map and a corresponding RGB image.
        print("Capturing an image from the depth camera ...")
        depth_image = capture_image(depth_cam)
        print("Capturing an image from the rgb camera ...")
        rgb_image = capture_image(rgb_cam)

        save(depth_image, rgb_image)

    setup_gui(capture, depth_cam, rgb_cam)

    # Free up the cameras.
    depth_cam.release()
    rgb_cam.release()

if __name__ == '__main__':
    main()
