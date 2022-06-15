import numpy as np
import fullscreen.fullscreen as fs

CAMERA_RESOLUTION = (1920, 1080)
VISUALIZED_PIXEL_SIZE = 8

def main():
    # Load the corners information from a file.
    with open("visualizations/pro_corners.npy", "rb") as corners_file:
        corners = np.load(corners_file)

    # Remove the extra middle dimension as in n x 1 x 2
    corner_count = corners.shape[0]
    corners = np.reshape(corners, (corner_count, 2))

    print(corners)

    cam_width, cam_height = CAMERA_RESOLUTION
    half_size = VISUALIZED_PIXEL_SIZE // 2

    # Make the corner pixels black and the others white
    pattern = np.ones((cam_height, cam_width, 3), dtype=np.uint8) * 255
    for i in range(corner_count):
        x, y = corners[i]
        for dx in range(-half_size, half_size):
            for dy in range(-half_size, half_size):
                X, Y = int(x) + dx, int(y) + dy
                if X >= 0 and X < cam_width and Y >= 0 and Y < cam_height:
                    pattern[Y, X, :] = np.array([0, 0, 0])
    
    # Project the pattern
    screen = fs.FullScreen("cv2", screen_id=1)
    screen.imshow(pattern)

    # Wait for the user to stop projecting the pattern
    print("Press Enter to stop the projection.")
    input()

    screen.destroyWindow()

if __name__ == '__main__':
    main()
