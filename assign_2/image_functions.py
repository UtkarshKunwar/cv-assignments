#import module
import numpy as np
import cv2
import sys, os
import argparse

#function to get pixel values from the given image using mouse pointer
def getPixels(event,x,y,flags,param):
    pts = np.zeros(1,1)
    print(pts)
    if event == cv2.EVENT_FLAG_LBUTTON:
        print(x,y)