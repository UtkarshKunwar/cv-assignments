# Import modules.
import numpy as np
import cv2
import sys, os
import argparse

def getPointCorrespondences(img1, img2, num=0):
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    if num != 0:
        return np.int32(pts1[:num]), np.int32(pts2[:num])
    return np.int32(pts1), np.int32(pts2)

# Gives the fundamental matrix.
def getFundamentalMatrix(pts1, pts2):

    #define A matrix
    A = np.zeros((len(pts1), 9))
    for i in range(len(pts1)):
        A.itemset((i, 0), pts1[i][0] * pts2[i][0])
        A.itemset((i, 1), pts1[i][1] * pts2[i][0])
        A.itemset((i, 2), pts2[i][0])
        A.itemset((i, 3), pts1[i][0] * pts2[i][1])
        A.itemset((i, 4), pts1[i][1] * pts2[i][1])
        A.itemset((i, 5), pts2[i][1])
        A.itemset((i, 6), pts1[i][0])
        A.itemset((i, 7), pts1[i][1])
        A.itemset((i, 8), 1)

    # get F (fundamental matrix using SVD)
    u, s, vh = np.linalg.svd(A, full_matrices=True)

    # F is the last column of v
    F = vh[:, 2]
    F = np.reshape(F, (3, 3))

    # decompose F using SVD
    u_f, s_f, vh_f = u, s, vh = np.linalg.svd(F, full_matrices=True)

    #make the last diagonal element of s_f to be zero
    s_f = np.diag(s_f)
    s_f.itemset((2, 2), 0)

    #multiply u_f, modified s_f, vh_f together to get the low rank F
    F = np.matmul(u_f, np.matmul(s_f, vh_f))

    return F
