# Import modules.
import numpy as np
import cv2
import sys, os
import argparse

# Returns the point correspondences between two images.
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

# Returns the normalised points along with the transformation matrix.
def getNormalizedCoordinates(pts):
    # Finding mean.
    avg = np.array([0, 0], dtype=np.float32)
    for pt in pts:
        avg[0] += pt[0]
        avg[1] += pt[1]
    avg = avg / len(pts)

    # Using norm to get scale.
    scale = 0
    for pt in pts:
        scale += np.linalg.norm([(pt[0] - avg[0]), (pt[1] - avg[1])])
    scale /= len(pts)
    scale = np.sqrt(2) / scale

    # Scaling points
    for pt in pts:
        pt[0] = (pt[0] - avg[0]) * scale
        pt[1] = (pt[1] - avg[1]) * scale

    # Transformation matrix.
    T = np.array([[scale, 0, -scale * avg[0]], [0, scale, -scale * avg[1]], [0, 0, 1]], dtype=np.float32)
    return pts, T

# Gives the fundamental matrix.
def getFundamentalMatrix(pts1, pts2):
    pts1_norm = np.copy(pts1)
    pts1_norm = np.float32(pts1_norm)
    pts1_norm, T1 = getNormalizedCoordinates(pts1_norm)
    pts2_norm = np.copy(pts2)
    pts2_norm = np.float32(pts2_norm)
    pts2_norm, T2 = getNormalizedCoordinates(pts2_norm)

    # Define A matrix
    A = np.zeros((len(pts1), 9))
    for i in range(len(pts1)):
        A.itemset((i, 0), pts1_norm[i][0] * pts2_norm[i][0])
        A.itemset((i, 1), pts1_norm[i][1] * pts2_norm[i][0])
        A.itemset((i, 2), pts2_norm[i][0])
        A.itemset((i, 3), pts1_norm[i][0] * pts2_norm[i][1])
        A.itemset((i, 4), pts1_norm[i][1] * pts2_norm[i][1])
        A.itemset((i, 5), pts2_norm[i][1])
        A.itemset((i, 6), pts1_norm[i][0])
        A.itemset((i, 7), pts1_norm[i][1])
        A.itemset((i, 8), 1)

    # Get F (fundamental matrix using eigenvalue).
    w, v = np.linalg.eig(np.matmul(np.transpose(A), A))
    F = v[:, len(v) - 1]

    # F is the last column of v.
    F = np.reshape(F, (3, 3))

    # Decompose F using SVD.
    u_f, s_f, vh_f = u, s, vh = np.linalg.svd(F, full_matrices=True)

    # Make the last diagonal element of s_f to be zero.
    s_f[2] = 0.0

    # Multiply u_f, modified s_f, vh_f together to get the low rank F.
    F = np.matmul(u_f, np.matmul(np.diag(s_f), vh_f))

    # Get unnormalised F.
    F = np.matmul(np.transpose(T2), np.matmul(F, T1))

    # Make last value 1
    F = F / F[2][2]

    return F
