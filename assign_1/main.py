# python program to read images and perform Rotation, Affine, Homography transformations on it

#numpy module
import numpy as np
# importing OpenCV(cv2) module
import cv2
# import user-defined image functions
import image_functions
import sys, os
import argparse

parser = argparse.ArgumentParser(description="CV Assignment 1 - Rotation, Affine, Homography")
parser.add_argument("img_path", help="Path for the image")
parser.add_argument("-o", "--origin", type=float, nargs=2, default=[0, 0], help="Origin for performing transformations.")
parser.add_argument("-r", "--rotation", type=float, help="Rotate the image by said degrees")
parser.add_argument("-rb", "--rotation-bilinear", type=float, help="Rotate the image by said degrees using bilinear interpolation.")
parser.add_argument("-a", "--affine", type=float, nargs=6, help="Perform affine using given parameters.")
parser.add_argument("-ho", "--homography", type=float, nargs=9, help="Perform homography using given parameters.")

args = parser.parse_args()
#------------------------------------------------------------------------------------------------------
# load the rgb image
img_path = args.img_path
if not os.path.exists(img_path):
    print("Invalid image path.")
    sys.exit(1)

img = cv2.imread(img_path)

cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original Image', 500, 500)
cv2.imshow('Original Image', img)

origin = np.array([[args.origin[0]], [args.origin[1]]])

#carry out rotation
if args.rotation:
    img_rot = image_functions.rotateImage(args.rotation, img, origin)
    cv2.namedWindow('Rotated Image', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Rotated Image', 500, 500)
    cv2.imshow('Rotated Image', img_rot)

if args.rotation_bilinear:
    img_rot_bi = image_functions.rotateImageBI(args.rotation_bilinear, img, origin)
    cv2.namedWindow('Rotated Image Bilinear', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Rotated Image Bilinear', 500, 500)
    cv2.imshow('Rotated Image Bilinear', img_rot_bi)

#carry out affine
#A = np.array([[1, 0.2, 1], [1, 0.5, 3]])
#A = np.array([[1, 0.2, 1], [1, 0.5, 3]])
if args.affine:
    A = np.array([[args.affine[0], args.affine[1], args.affine[2]],
                  [args.affine[3], args.affine[4], args.affine[5]]])
    img_aff = image_functions.affine(A, img, origin)
    cv2.namedWindow('Affined Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Affined Image', 500, 500)
    cv2.imshow('Affined Image', img_aff)

#carry out homography
#H = np.array([[1, 2, 3], [4, 3, 9], [0.001, 0.01, 1]])
#H = np.array([[1, 2, 3], [4, 3, 9], [0.001, 0.01, 10]])

if args.homography:
    H = np.array([[args.homography[0], args.homography[1], args.homography[2]],
                  [args.homography[3], args.homography[4], args.homography[5]],
                  [args.homography[6], args.homography[7], args.homography[8]]])
    img_homo = image_functions.homography(H, img, origin)
    cv2.namedWindow('Homographied Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Homographied Image', 500, 500)
    cv2.imshow('Homographied Image', img_homo)

# Maintain output window until
# user presses a key
cv2.waitKey(0)

# Destroying present windows on screen
cv2.destroyAllWindows()
#------------------------------------------------------------------------------------------------------
