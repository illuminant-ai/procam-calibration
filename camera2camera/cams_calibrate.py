#!/usr/bin/env python3

import glob
import cv2
import os.path
import numpy as np
import json

CHESS_SHAPE = (10, 7)
IMAGE_SIZE = (1920, 1080)

def get_valid_dirs():
    """
    Fetches the list of valid capture directories. A directory is valid if it 
    is indexed and contains a depth.png and rgb.png file.
    """
    dirs = sorted(glob.glob("./captures/capture_*"), key=lambda d: int(d.split("_")[1]))
    valid_dirs = []
    for dir in dirs:
        if os.path.exists(f"{dir}/depth.png") and \
            os.path.exists(f"{dir}/rgb.png"):
            valid_dirs.append(dir)
    return valid_dirs

def read_images(dir):
    """
    Reads the two images, depth.png and rgb.png, from the directory. Assumes
    the directory is a valid capture directory.
    """
    depth = cv2.imread(f"{dir}/depth.png", cv2.IMREAD_GRAYSCALE)
    rgb = cv2.imread(f"{dir}/rgb.png", cv2.IMREAD_GRAYSCALE)
    return depth, rgb


def visualize_corners(image, corners, label=""):
    """
    Displays a visualization of the corners on the image. The visualization
    pauses the program and is displayed until the user presses a key.
    """
    colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(colored_image, CHESS_SHAPE, corners, True)
    cv2.imshow(label, cv2.pyrDown(colored_image))
    cv2.waitKey(0)
    cv2.destroyWindow(label)

def load_camera_params(config_file):
    """
    Parses the intrinsic matrix and the distortion values of the camera from
    a camera configuration file in JSON format, and returns them as a tuple.
    """
    with open(config_file, 'r') as file:
        param_data = json.load(file)
        P = param_data['camera']['P']
        D = param_data['camera']['distortion']
        return np.array(P).reshape([3,3]), np.array(D)

def main():
    """
    The program does the following in order:

    1. Extract the intrinsic matrix of the rgb camera using the calculated
       chessboard corners.
    2. Load in the intrinsic matrix of the depth camera.
    3. Find the extrinsics between the two cameras using stereo calibration.
    4. Save the intrinsics and extrinsics to a results file.
    5. Compares the results across multiple poses and check if they converge.
       If they converge, then the calibration process went smoothly.
    """
    cal_points = np.zeros((CHESS_SHAPE[0] * CHESS_SHAPE[1], 3), np.float32)
    cal_points[:, :2] = np.mgrid[:CHESS_SHAPE[0], :CHESS_SHAPE[1]].T.reshape(-1, 2)

    depth_corners_ls, rgb_corners_ls = [], []

    valid_dirs = get_valid_dirs()
    for dir in valid_dirs:
        print(f"Processing capture directory {os.path.normcase(dir)}\\")
        depth_image, rgb_image = read_images(dir)

        # Find the chessboard corners in both images.
        depth_found, depth_corners = cv2.findChessboardCorners(depth_image, CHESS_SHAPE)
        rgb_found, rgb_corners = cv2.findChessboardCorners(rgb_image, CHESS_SHAPE)

        # If the chessboard is not visible, skip the capture.
        if not depth_found or not rgb_found:
            print(f"SKIP: Chessboard not found in {os.path.normcase(dir)}")
            continue

        # Add the corners to the list of corners for each image.
        depth_corners_ls.append(depth_corners)
        rgb_corners_ls.append(rgb_corners)
    
        # Visualize the chessboard corners.
        visualize_corners(depth_image, depth_corners, f"Annotated {dir}\\depth")
        visualize_corners(rgb_image, rgb_corners, f"Annotated {dir}\\rgb")

    processed_dir_count = len(depth_corners_ls)
    cal_points_ls = [cal_points] * processed_dir_count

    # Load in the intrinsics of the depth camera.
    depth_int, depth_dist = load_camera_params("./camera_config.json")

    # Calibrate or find the intrinsics of the rgb camera independently.
    flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT
    rgb_rms, rgb_int, rgb_dist, _, _, _, _, rgb_rms_per_view = \
        cv2.calibrateCameraExtended(cal_points_ls, rgb_corners_ls, \
            IMAGE_SIZE, None, None, flags=flags)
        
    # Calibrate the stereo system.
    print("Calibrating the stereo system ...")
    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    rms, _, _, _, _, R, T, E, F, rms_per_view = cv2.stereoCalibrateExtended(
        cal_points_ls, depth_corners_ls, rgb_corners_ls, depth_int, depth_dist, \
            rgb_int, rgb_dist, IMAGE_SIZE, None, None, stereo_flags
    )
    
    # Log the results into a results file.
    print("Logging the calibration results into calibration_result.json ...")
    fs = cv2.FileStorage('calibration_result.json', cv2.FILE_STORAGE_WRITE)
    fs.write("RGB camera RMS", rgb_rms)                 # RGB
    fs.write("RGB camera RMS per view", rgb_rms_per_view)
    fs.write("RGB camera intrinsics", rgb_int)
    fs.write("RGB camera distortions", rgb_dist)
    fs.write("Stereo RMS", rms)                         # Stereo
    fs.write("Stereo RMS per view", rms_per_view)
    fs.write("Stereo Rotation Matrix", R)
    fs.write("Stereo Translation Matrix", T)
    fs.release()

if __name__ == "__main__":
    main()
