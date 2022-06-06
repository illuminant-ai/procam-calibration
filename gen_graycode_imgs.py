#coding: UTF-8

import os
import os.path
import argparse
import cv2
import numpy as np
import math

TARGETDIR = './graycode_pattern'
CAPTUREDDIR = './capture_*'

def main():
    # Setup command line argument parsing.
    #
    # graycode_step controls how many pixels the finest strip of the graycode spans.
    #   At least, in the context of range(depth) finding using structured light,
    #   a larger graycode_step implies a lower resolution in the resulting depth map.
    parser = argparse.ArgumentParser(
        description='Generate graycode pattern images')
    parser.add_argument('proj_height', type=int, help='projector pixel height')
    parser.add_argument('proj_width', type=int, help='projector pixel width')
    parser.add_argument('-graycode_step', type=int,
                        default=1, help='step size of graycode [default:1](increase if moire appears)')
    args = parser.parse_args()
    
    # Extract the command line arguments.
    step = args.graycode_step
    height = args.proj_height
    width = args.proj_width

    # The graycode dimensions are downsampled by a factor of step.
    gc_height = math.ceil(height/step)      ## Original: int((height - 1)/step) + 1
    gc_width = math.ceil(width/step)

    # Create the graycode patterns with the reduced dimensions.
    graycode = cv2.structured_light.GrayCodePattern_create(gc_width, gc_height)     ## complies to documentation
    _, patterns = graycode.generate()       ## More readable

    # Upsample the downsampled graycode patterns to match the dimensions of the projector.
    exp_patterns = []
    for pat in patterns:
        img = np.zeros((height, width), np.uint8)
        for y in range(height):
            for x in range(width):
                img[y, x] = pat[int(y/step), int(x/step)]
        exp_patterns.append(img)

    # Add a white and black image to the list of graycode patterns.
    exp_patterns.append(255*np.ones((height, width), np.uint8))     # white
    exp_patterns.append(np.zeros((height, width), np.uint8))        # black

    # Make the target directory for storing graycode patterns if it doesn't exist.
    if not os.path.exists(TARGETDIR):
        os.mkdir(TARGETDIR)

    # Store the graycode patterns in the directory in the format 'pattern_<index>.png'.
    for i, pat in enumerate(exp_patterns):
        cv2.imwrite(TARGETDIR + '/pattern_' + str(i).zfill(2) + '.png', pat)

    print('=== Result ===')
    print('\'' + TARGETDIR + '/pattern_00.png ~ pattern_' +
          str(len(exp_patterns)-1) + '.png \' were generated')
    print()
    print('=== Next step ===')
    print('Project patterns and save captured images as \'' +
          CAPTUREDDIR + '/graycode_*.png\'')
    print()
    print(
        '    ./ --- capture_1/ --- graycode_00.png\n'
        '        |              |- graycode_01.png\n'
        '        |              |        .\n'
        '        |              |        .\n'
        '        |              |- graycode_' +
        str(len(exp_patterns)-1) + '.png\n'
        '        |- capture_2/ --- graycode_00.png\n'
        '        |              |- graycode_01.png\n'
        '        |      .       |        .\n'
        '        |      .       |        .\n')
    print()
    print('It is recommended to capture more than 5 times')
    print()


if __name__ == '__main__':
    main()
