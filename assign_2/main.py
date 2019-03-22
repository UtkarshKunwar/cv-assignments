# Program for assignment 2
#import module
import numpy as np
import cv2
import image_functions
import sys, os
import argparse
import imutils
from imutils import paths
from matplotlib import pyplot as plt

#----------------------------------------------------------------------------------
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
	help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True,
	help="path to the output image")
args = vars(ap.parse_args())

# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])))
images = []

img1 = cv2.imread(imagePaths[0])
img2 = cv2.imread(imagePaths[1])

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(img1),plt.title('Image 1')
ax2.imshow(img2),plt.title('Image 2')

cid = fig.canvas.mpl_connect('button_press_event', image_functions.onclick)

plt.show()
