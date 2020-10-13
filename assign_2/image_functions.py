#import module
import numpy as np
import cv2
import sys, os
import argparse

#form matrix A and B for AX = B using given 8 points
def matrixFormation(pts1, pts2):

    B = np.array(pts2)
    B = np.reshape(B, 8)
    B = np.transpose(B)

    A = np.zeros((8,8))
    rows = [0,2,4,6] #to iterate for 4 points
    counter = 0

    for i in rows:

        #set row for x coordinate
        A.itemset((i,0), pts1[counter][0])
        A.itemset((i,1), pts1[counter][1])
        A.itemset((i,2), 1)
        A.itemset((i,3), 0)
        A.itemset((i,4), 0)
        A.itemset((i,5), 0)
        A.itemset((i,6), pts1[counter][0] * pts2[counter][0] * -1)
        A.itemset((i,7), pts1[counter][1] * pts2[counter][0] * -1)

        # set row for y coordinate
        A.itemset((i+1, 0), 0)
        A.itemset((i+1, 1), 0)
        A.itemset((i+1, 2), 0)
        A.itemset((i+1, 3), pts1[counter][0])
        A.itemset((i+1, 4), pts1[counter][1])
        A.itemset((i+1, 5), 1)
        A.itemset((i+1, 6), pts1[counter][0] * pts2[counter][1] * -1)
        A.itemset((i+1, 7), pts1[counter][1] * pts2[counter][1] * -1)

        counter = counter + 1

    return A, B

# gives the fundamental matrix
def getFundamental(pts1, pts2):

    #define A matrix
    A = np.zeros((8, 9))
    for i in range(9):
        A.itemset((i, 0), pts1[i][0] * pts2[i][0])
        A.itemset((i, 1), pts1[i][1] * pts2[i][0])
        A.itemset((i, 2), pts2[i][0])
        A.itemset((i, 3), pts1[i][0] * pts2[i][1])
        A.itemset((i, 4), pts1[i][1] * pts2[i][1])
        A.itemset((i, 5), pts2[i][1])
        A.itemset((i, 6), pts1[i][0])
        A.itemset((i, 7), pts1[i][1])
        A.itemset((i, 8), 1)

    #get F (fundamental matrix using SVD)
    u, s, vh = np.linalg.svd(A, full_matrices=True)

    #F is the last column of v

    #decompose F using SVD
    u_f, s_f, vh_f = u, s, vh = np.linalg.svd(F, full_matrices=True)

    #make the last diagonal element of s_f to be zero

    #multiply u_f, modified s_f, vh_f together to get the low rank F

    return F