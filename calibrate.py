# coding: UTF-8

import math
import os
import os.path
import glob
import argparse
import cv2
import numpy as np
import json


def main():
    # Set up command line argument parsing.
    parser = argparse.ArgumentParser(
        description='Calibrate pro-cam system using chessboard and structured light projection\n'
        '  Place captured images as \n'
        '    ./ --- capture_1/ --- graycode_00.png\n'
        '        |              |- graycode_01.png\n'
        '        |              |        .\n'
        '        |              |        .\n'
        '        |              |- graycode_??.png\n'
        '        |- capture_2/ --- graycode_00.png\n'
        '        |              |- graycode_01.png\n'
        '        |      .       |        .\n'
        '        |      .       |        .\n',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('proj_height', type=int, help='projector pixel height')
    parser.add_argument('proj_width', type=int, help='projector pixel width')
    parser.add_argument('chess_vert', type=int,
                        help='number of cross points of chessboard in vertical direction')
    parser.add_argument('chess_hori', type=int,
                        help='number of cross points of chessboard in horizontal direction')
    parser.add_argument('chess_block_size', type=float,
                        help='size of blocks of chessboard (mm or cm or m)')
    parser.add_argument('graycode_step', type=int,
                        default=1, help='step size of graycode')
    parser.add_argument('-black_thr', type=int, default=40,
                        help='threshold to determine whether a camera pixel captures projected area or not (default : 40)')
    parser.add_argument('-white_thr', type=int, default=5,
                        help='threshold to specify robustness of graycode decoding (default : 5)')
    parser.add_argument('-camera', type=str, default=str(),help='camera internal parameter json file')

    # Extract the command line arguments into variables.
    args = parser.parse_args()
    proj_shape = (args.proj_height, args.proj_width)        # shape = (height, width)
    chess_shape = (args.chess_vert, args.chess_hori)
    chess_block_size = args.chess_block_size
    gc_step = args.graycode_step
    black_thr = args.black_thr
    white_thr = args.white_thr
    camera_param_file = args.camera

    # Make sure there is at least one capture directory.
    dirnames = sorted(glob.glob('./capture_*'))
    if len(dirnames) == 0:
        print('Directories \'./capture_*\' were not found')
        return
    ## Suggestion: assert len(dirnames) > 0, "Directories './capture_*' were not found."

    # Store non-empty capture directories and the graycode_* files in these directories
    # in the lists used_dirnames and gc_fname_lists.
    #   Note: gc_fname_lists is a list of lists containing the graycode_* files, NOT a
    #         one-dimensional list containing all the graycode_* files.
    print('Searching input files ...')
    used_dirnames = []
    gc_fname_lists = []
    for dname in dirnames:
        gc_fnames = sorted(glob.glob(dname + '/graycode_*'))
        if len(gc_fnames) == 0:
            continue
        used_dirnames.append(dname)
        gc_fname_lists.append(gc_fnames)
        print(' \'' + dname + '\' was found')

    # Load the camera parameters which are the intrinsic matrix and the distortions.
    camP = None             # Camera Intrinsic Matrix
    cam_dist = None         # Camera Distortions
    _, ext = os.path.splitext(camera_param_file)
    if ext == ".json":
        camP, cam_dist = loadCameraParam(camera_param_file)
        print('loading camera parameters')
        print(camP)
        print(cam_dist)

    # Calibrate the camera-projector system
    calibrate(used_dirnames, gc_fname_lists,
              proj_shape, chess_shape, chess_block_size, gc_step, black_thr, white_thr,
              camP, cam_dist)


def printNumpyWithIndent(tar, indentchar):
    print(indentchar + str(tar).replace('\n', '\n' + indentchar))

def loadCameraParam(json_file):
    """
    Parses the intrinsic matrix and the distortion values of the camera from
    a camera configuration file in JSON format, and returns them as a tuple.
    """
    with open(json_file, 'r') as f:
        param_data = json.load(f)
        P = param_data['camera']['P']
        d = param_data['camera']['distortion']
        return np.array(P).reshape([3,3]), np.array(d)

def calibrate(dirnames, gc_fname_lists, proj_shape, chess_shape, chess_block_size, gc_step, black_thr, white_thr, camP, camD):
    # objps represent the calibration pattern points in the calibration pattern coordinate space.
    # Refer to https://docs.opencv.org/4.4.0/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d.
    objps = np.zeros((chess_shape[0]*chess_shape[1], 3), np.float32)
    objps[:, :2] = chess_block_size * \
        np.mgrid[:chess_shape[0], :chess_shape[1]].T.reshape(-1, 2)

    print('Calibrating ...')

    # Create a downsampled graycode pattern generator parameterized by gc_step.
    gc_height = math.ceil(proj_shape[0]/gc_step)    ## Original: gc_height = int((proj_shape[0]-1)/gc_step)+1
    gc_width = math.ceil(proj_shape[1]/gc_step)     ## Original: gc_width = int((proj_shape[1]-1)/gc_step)+1
    graycode = cv2.structured_light.GrayCodePattern_create(gc_width, gc_height) ## complies to documentation
    graycode.setBlackThreshold(black_thr)
    graycode.setWhiteThreshold(white_thr)

    # Find the dimensions of the camera image plane from one of the graycode_* files.
    cam_shape = cv2.imread(gc_fname_lists[0][0], cv2.IMREAD_GRAYSCALE).shape

    # Set the patch size according to a formula involving the width of the camera dimension.
    # Comments: The patch size is a hyperparameter. If it's too small, it's sensitive to decoding
    #           errors. If it's too large, it's robust to errors but unable to cope with lens
    #           distortions. (Moreno et al.)    ## FIX
    patch_size_half = int(np.ceil(cam_shape[1] / 15))       # = 1920 / 15 = 128
    print('  patch size :', patch_size_half * 2)            # = 128 * 2 = 256

    # Create two lists of lists of corners, one in calibration pattern coordinates
    # (corners) and one in the coordinates of the image plane (objps), for the camera
    # and the projector.
    cam_corners_list = []
    cam_objps_list = []
    cam_corners_list2 = []
    proj_objps_list = []
    proj_corners_list = []

    # Loop through each directory containing the graycode-pattern projected images.
    # Each directory is associated with a different pose of the image.
    for dname, gc_filenames in zip(dirnames, gc_fname_lists):
        dname_index = dname.split("_")[1]
        print('  checking \'' + dname + '\'')

        # Assert the graycode_step parameter passed to the cap_chessboard.py equals the
        # one passed to this program.
        ## Suggestion: An assert statement.
        if len(gc_filenames) != graycode.getNumberOfPatternImages() + 2:
            print('Error : invalid number of images in \'' + dname + '\'')
            return None

        # Transform the graycode-projected images into grayscale images.
        projected_graycodes = []
        for fname in gc_filenames:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            # Ensure that the camera dimensions agree with the size of the images.
            ## Suggestion: An assert statement.
            if cam_shape != img.shape:
                print('Error : image size of \'' + fname + '\' is mismatch')
                return None
            projected_graycodes.append(img)

        # Extract the black and white projected images from the list into separate variables.
        black_img = projected_graycodes.pop()
        white_img = projected_graycodes.pop()

        # Find the coordinates of the inner chessboard corners in the image of the 
        # camera by using a clear (black-projected) image of a chessboard.
        #   Note: findChessboardCorners returns a list of dimension n x 1 x 2 where
        #         n is the number of corners, instead of the simpler dimension n x 2.
        res, cam_corners = cv2.findChessboardCorners(black_img, chess_shape)
        if not res:
            print('Error : chessboard was not found in \'' +
                  gc_filenames[-2] + '\'')
            return None
        cam_corners_list.append(cam_corners)

        # Append the same fixed camera calibration points into the objps_list. We need to
        # change this if we are using different calibration patterns or occluding some
        # points for the calibration.
        # For now, we are only using a single chessboard pattern, so the visible
        # calibration points are always the same.
        cam_objps_list.append(objps)

        proj_objps = []
        proj_corners = []
        cam_corners2 = []
        viz_cam_points = cv2.cvtColor(white_img, cv2.COLOR_GRAY2RGB)

        # Map each camera pixel to a projector pixel using the projected structured light
        # patterns (which are converted to grayscale). The resulting image is used ONLY
        # for visualization purposes. This visualization allows us to see the projector
        # coordinates from the camera's image plane. A perfect mapping algorithm should
        # produce a visualization that is a perfect gradient from green (lower-left) to
        # blue (upper-right), assuming the camera and the projector are coaxially aligned.
        # In reality, there are slight rotations as well as inaccuracies in the image due
        # to the orientation of the projector.
        #   The purpose of adding the brightness value of the image as the third
        #   component is to project the image onto this color mapping.
        image = np.zeros((cam_shape[0], cam_shape[1], 3), dtype=np.uint16)
        for row in range(0, cam_shape[0]):
            for col in range(0, cam_shape[1]):
                ret, pp = graycode.getProjPixel(projected_graycodes, col, row)
                # Normalize the projector pixels with the camera dimensions, because we
                # are thinking of the projector dimensions and the camera dimensions as
                # being the same in this context. In an ideal setting, there is a bijective
                # mapping between the camera pixel and the projector pixels. This is not
                # exactly so in reality, but this model is useful.
                image[row, col, 0] = (pp[0] / cam_shape[1]) * 2**16
                image[row, col, 1] = (pp[1] / cam_shape[0]) * 2**16
                # Store the intensity of the grayscale pixel in the 3rd entry.
                image[row, col, 2] = (black_img[row, col] / 255) * 2**16
        cv2.imwrite('viz_pro_gc_' + str(dname_index) + '.png', image)

        # Loop through each inner chessboard corner in terms of its coordinate in the
        # image plane and the calibration pattern coordinate plane.
        for corner, objp in zip(cam_corners, objps):
            c_x = int(round(corner[0][0]))
            c_y = int(round(corner[0][1]))
            src_points = []
            dst_points = []

            # Loop through each pixel in the patch, the center of which is an inner
            # corner of the chessboard.
            for dx in range(-patch_size_half, patch_size_half + 1):
                for dy in range(-patch_size_half, patch_size_half + 1):
                    x = c_x + dx
                    y = c_y + dy

                    # Check that the patch pixel under consideration is not out of range.
                    if (y < 0 or y >= cam_shape[0] or x < 0 or x >= cam_shape[1]):
                        continue

                    # The black threshold determines the minimum brightness difference
                    # between the fully illuminated (white projection) image and the
                    # unilluminated (black projection) image, required for a valid pixel.
                    # If the difference is smaller than this threshold, then the pixel
                    # is considered invalid.
                    # Refer to https://docs.opencv.org/4.4.0/d1/dec/classcv_1_1structured__light_1_1GrayCodePattern.html#a3607cba801a696881df31bbd6c59c4fd.
                    #
                    # PROBLEM: Why is this here? Is it to prune out invalid pixels as
                    # stated in the documentation of setBlackThreshold()? What exactly are
                    # these thresholds used for?
                    if int(white_img[y, x]) - int(black_img[y, x]) <= black_thr:
                        continue

                    # If there is no error finding the projector pixel, append these to
                    # arrays that together describe a mapping. The reason why there is a
                    # multiplication of a gc_step can be reasoned as follows. The
                    # 'graycode.getProjPixel' infers which projector pixel each camera pixel
                    # maps to, using the structured light patterns. More specifically, it
                    # uses the highest frequency one to determine the coordinates of the pixel.
                    # gc_step controls the size of this highest-frequency pattern and the higher
                    # it is, the larger this strip and the area of the pixel inferred by
                    # graycode.getProjPixel. A larger pixel area means the pixel coordinates
                    # will be scaled down accordingly, since we are working the same total area.
                    # Muliplying with gc_step normalizes these coordinates back to match
                    # the true pixel dimensions of the projector, in contrast to the pixel
                    # dimensions inferred from the graycode patterns.
                    err, proj_pix = graycode.getProjPixel(projected_graycodes, x, y)
                    if not err:
                        src_points.append((x, y))
                        dst_points.append(gc_step*np.array(proj_pix))

            # If the number of decodable pixels is less than a quarter the area of the
            # patch size, we skip further processing on this corner.
            if len(src_points) < patch_size_half**2:
                print(
                    '    Warning : corner', c_x, c_y,
                    'was skipped because decoded pixels were too few (check your images and thresholds)')
                continue

            # Find a homography based on the decodable pixels in the patch -- since this
            # homography is computed based on a patch localized to the corner, this is
            # termed a local homography.
            h_mat, inliers = cv2.findHomography(
                np.array(src_points), np.array(dst_points))

            # Use the local homography to find the coordinates of the corner in the
            # projector image plane.
            point = h_mat@np.array([corner[0][0], corner[0][1], 1]).transpose()     ## .transpose() is unnecessary
            point_pix = point[0:2]/point[2]         # PROBLEM: Re-transform into non-homogenous coordinates?
            proj_objps.append(objp)
            proj_corners.append([point_pix])        # complies with the shape of cam_corners (n x 1 x 2)

            # Append the corners, for which homographies can be calculated, into a list.
            cam_corners2.append(corner)

        # Assert that at least three corners in the projector coordinate system are
        # able to be calculated.
        if len(proj_corners) < 3:
            print('Error : too few corners were found in \'' + dname + '\' (less than 3)')
            return None

        # Store the coordinates of the calculatable inner corners in terms of calibration
        # pattern coordinates, projector image plane coordinates and the camera image
        # plane coordinates.
        proj_objps_list.append(np.float32(proj_objps))
        proj_corners_list.append(np.float32(proj_corners))
        cam_corners_list2.append(np.float32(cam_corners2))

        # Create the image plane of the projector from that of the camera using a
        # homography. As Phil correctly mentioned, this homography matrix used
        # is the most recent local homography computed. Since the transformation is
        # global, a global homography encompassing the entire image plane is more
        # appropriate and should produce better results. This use of a local homography
        # matrix for a global transformation is likely responsible for the inaccurate
        # corner markings in the viz_pro_corners_* files.       ## FIX
        viz_pro_points = cv2.warpPerspective(viz_cam_points, h_mat, dsize=(1920, 1080))     ## Clearer

        print(cam_corners.shape)

        # Draw the chessboard corners in the camera and projector image planes and save
        # these marked image planes.
        cv2.drawChessboardCorners(viz_cam_points, chess_shape, cam_corners, True)
        cv2.drawChessboardCorners(viz_pro_points, chess_shape, np.float32(proj_corners), True)
        cv2.imwrite('viz_cam_corners_' + str(dname_index) + '.png', viz_cam_points)
        cv2.imwrite('viz_pro_corners_' + str(dname_index) + '.png', viz_pro_points)

    print('Initial solution of camera\'s intrinsic parameters')

    # Perform camera calibration based on the coordinates of the inner chessboard
    # corners. The calibration methods differ based on whether the camera's parameters,
    # which are its intrinsic matrix and the distortions, are known beforehand.
    #   Note: For both methods, the rotation matrices and translation vectors are
    #         computed for each pose. On the other hand, the intrinsic matrix and the
    #         distortions are computed based on all poses.
    cam_rvecs = []
    cam_tvecs = []
    if camP is None:
        # Currently, we are fixing the principal point of the camera, as well as the
        # distortions and the aspect ratio. The aspect ratio is perhaps meant to be
        # fixed, but the principal point should NOT be fixed.   ## FIX
        cam_flags = cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2| cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT
        ret, cam_int, cam_dist, cam_rvecs, cam_tvecs, _, _, cam_per_view_errors = cv2.calibrateCameraExtended(cam_objps_list, cam_corners_list, (1920, 1080), None, None, flags=cam_flags)
        print('Camera Shape: ', cam_shape)
        print('  RMS :', ret)
        print('  RMS Per View:', cam_per_view_errors)
    else:
        for objp, corners in zip(cam_objps_list, cam_corners_list):
            ret, cam_rvec, cam_tvec = cv2.solvePnP(objp, corners, camP, camD) 
            cam_rvecs.append(cam_rvec)
            cam_tvecs.append(cam_tvec)
            print('  RMS :', ret)
        cam_int = camP
        cam_dist = camD
    print('  Intrinsic parameters :')
    printNumpyWithIndent(cam_int, '    ')
    print('  Distortion parameters :')
    printNumpyWithIndent(cam_dist, '    ')
    print()

    print('Initial solution of projector\'s parameters')

    # Calibrate the projector in the same way we calibrate the camera, using the
    # coordinates of the inner chessboard corners.
    pro_flags = cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2| cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    """
    pro_flags = cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2| cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT
    pro_flags = cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2| cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_ASPECT_RATIO
    pro_flags = cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2| cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_PRINCIPAL_POINT
    """
    ret, proj_int, proj_dist, proj_rvecs, proj_tvecs, _, _, pro_per_view_errors= cv2.calibrateCameraExtended(proj_objps_list, proj_corners_list, (1920, 1080), None, None, flags=pro_flags)
    print('  RMS :', ret)
    print('  RMS :', pro_per_view_errors)
    print('  Intrinsic parameters :')
    printNumpyWithIndent(proj_int, '    ')
    print('  Distortion parameters :')
    printNumpyWithIndent(proj_dist, '    ')
    print()

    print('=== Result ===')

    # Calibrate the stereo camera/projector setup.
    ste_flags = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_USE_INTRINSIC_GUESS     ## FIX: The latter flag is useless, given the former. Also, since the documentation suggests fixing the intrinsics, if we have found them us ing cameraCalibrate, we have more motivation to remove the second flag that does the opposite.
    ret, cam_int, cam_dist, proj_int, proj_dist, cam_proj_rmat, cam_proj_tvec, E, F, ste_per_view_errors = cv2.stereoCalibrateExtended(proj_objps_list, cam_corners_list2, proj_corners_list, cam_int, cam_dist, proj_int, proj_dist, (1920, 1080), None, None, flags=ste_flags)     ## The image_size was previously unspecified (None), though not specifying it does not have any effect as the intrinsic matrices are already calculated. Refer to documentation here: https://docs.opencv.org/4.4.0/d9/d0c/group__calib3d.html#ga91018d80e2a93ade37539f01e6f07de5.

    print('  RMS :', ret)
    print('  RMS :', ste_per_view_errors)
    print('  Camera intrinsic parameters :')
    printNumpyWithIndent(cam_int, '    ')
    print('  Camera distortion parameters :')
    printNumpyWithIndent(cam_dist, '    ')
    print('  Projector intrinsic parameters :')
    printNumpyWithIndent(proj_int, '    ')
    print('  Projector distortion parameters :')
    printNumpyWithIndent(proj_dist, '    ')
    print('  Rotation matrix / translation vector from camera to projector')
    print('  (they translate points from camera coord to projector coord) :')
    printNumpyWithIndent(cam_proj_rmat, '    ')
    printNumpyWithIndent(cam_proj_tvec, '    ')
    print()

    # Store the solved parameters of the stereo system into a result file.
    fs = cv2.FileStorage('calibration_result.json', cv2.FILE_STORAGE_WRITE)
    fs.write('img_shape', cam_shape)
    fs.write('rms', ret)
    fs.write('cam_int', cam_int)
    fs.write('cam_dist', cam_dist)
    fs.write('proj_int', proj_int)
    fs.write('proj_dist', proj_dist)
    fs.write('rotation', cam_proj_rmat)
    fs.write('translation', cam_proj_tvec)
    fs.release()


if __name__ == '__main__':
    main()
