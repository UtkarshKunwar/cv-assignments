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
ap.add_argument("-i", "--images", type=str, required=True, help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True, help="path to the output image")
args = vars(ap.parse_args())

# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])))
images = []

#loading and swaping image channels for proper display in matplotlib
img1 = cv2.imread(imagePaths[0])
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread(imagePaths[1])
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
print('Image1 size: ',img1.shape,'Image 2 size: ', img2.shape)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(img1), ax1.set_title('Image 1')
ax2.imshow(img2), ax2.set_title('Image 2')

#function to get pixel values from the given image using mouse pointer using matplotlib
counter = 1
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    global coords
    global counter
    if counter == 1:
        coords[0].append((ix, iy))
    else:
        coords[1].append((ix, iy))
    counter = counter * -1

    if len(coords[1]) == 4:
        fig.canvas.mpl_disconnect(cid)
        plt.close(1)
    return

coords = [[], []]
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show(1)

pts1 = np.float32(coords[0])
pts2 = np.float32(coords[1])

M = cv2.getPerspectiveTransform(pts1,pts2)
print('M through open cv : \n', M)

dst = cv2.warpPerspective(img1,M,(1000,1000))
plt.imshow(dst)
plt.show(2)

A, B = image_functions.matrixFormation(pts1, pts2)
sol_ls = np.linalg.lstsq(A,B, rcond=None)
print('M through least square method: \n', sol_ls)