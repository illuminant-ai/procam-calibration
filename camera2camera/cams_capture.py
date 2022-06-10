import cv2
import glob
import os

CAMERA_RESOLUTION = (1920, 1080)
FLUSH_COUNT = 10

def flush(camera):
    """ Flushes the camera buffer to ready for the next capture. """
    for _ in range(FLUSH_COUNT):
        camera.grab()

def setup_camera(index, label=""):
    """ Fetches the camera with the specified index and sets it up. """
    camera = cv2.VideoCapture(index)
    # Tries to set the capture properties of the camera.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    # First and foremost flush before a camera capture.
    flush(camera)
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
        dirs = sorted(glob.glob("./captures/cap_*"), key=lambda x: int(x.split("_")[1]))
        recent = "./captures/cap_-1" if len(dirs) == 0 else dirs[-1]
        recent_index = int(recent.split("_")[1])
        next_index = recent_index + 1
        return f"./captures/cap_{next_index}"

    dir = get_next_directory()
    os.mkdir(dir)
    cv2.imwrite(f"{dir}/depth.png", depth_image)
    cv2.imwrite(f"{dir}/rgb.png", rgb_image)

def main():
    # ADJUST these indices to reflect the actual indices of the camera.
    print("Setting up the cameras for capture ...")
    depth_cam = setup_camera(2, "depth")
    rgb_cam = setup_camera(1, "rgb")

    # These does NOT imply a depth map and a corresponding RGB image.
    print("Capturing an image from the depth camera ...")
    depth_image = capture_image(depth_cam)
    print("Capturing an image from the rgb camera ...")
    rgb_image = capture_image(rgb_cam)

    print("Saving the images to a capture directory ...")
    save(depth_image, rgb_image)

    # Free up the cameras.
    depth_cam.release()
    rgb_cam.release()

if __name__ == '__main__':
    main()
