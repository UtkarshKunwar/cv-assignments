# Program for assignment 2
# Import modules
import numpy as np
import cv2
from image_functions import *
import sys, os
import argparse
import imutils
from imutils import paths
from matplotlib import pyplot as plt
np.set_printoptions(precision=4)
#----------------------------------------------------------------------------------
# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True, help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True, help="path to the output image")
args = vars(ap.parse_args())

# Grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])))
images = []

# Loading and swaping image channels for proper display in matplotlib
img1 = cv2.imread(imagePaths[0])
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread(imagePaths[1])
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
print('Image1 size: ',img1.shape,'Image 2 size: ', img2.shape)

pts1, pts2 = getPointCorrespondences(img1, img2)

# Find and print the fundamental matrix.
F = getFundamentalMatrix(pts1, pts2)
print("Fundamental Matrix:\n", F)

cv_F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
cv_F = np.asarray(cv_F, dtype=np.float32)
print("\nCV2 Fundamental Matrix:\n", cv_F)
