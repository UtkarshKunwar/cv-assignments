# file containing basic image functions such as
# rotation, affine and homography, etc.

import cv2
import numpy as np
import math

#-------------------------------------------------------------------------------------------------------------
# function to display images and kill windows after displaying it
def displayImage (window_name, img):
    cv2.imshow(window_name, img)
    # Maintain output window until
    # user presses a key
    cv2.waitKey(0)

    # Destroying present windows on screen
    cv2.destroyAllWindows()

#-------------------------------------------------------------------------------------------------------------
# function to rotate images by given theta degrees
def rotateImage (theta, img, origin):

    #convert angle to degrees
    angle = math.radians(theta)

    #get image dimenstions
    [rows, cols, chn] = img.shape

    #define new image size with padding
    img_aft_rot = np.zeros((2 * max(rows, cols), 2 * max(rows, cols), 3))
    print('New Image Size: ', img_aft_rot.shape)

    # define rotation matrix
    R = np.array([[math.cos(angle), -1 * math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    #print('Rotation Matrix: ', R)

    # define rotation + translation matrix. Basically origin is shifted
    T = np.append(R, origin, 1)
    #print('T matrix: ', T)

    for x in range(rows) :
        for y in range(cols) :

            #find new pixel coordinates corresponding to rotation
            cord_bfr_rot = np.array([[x],[y],[1]])
            cord_aft_rot = np.matmul(T, cord_bfr_rot)

            #round-off new pixel indices
            new_index = [math.ceil(cord_aft_rot.item(0)), math.ceil(cord_aft_rot.item(1))]

            for i in range(chn) :
                img_aft_rot.itemset((new_index[0], new_index[1], i), img.item(x, y, i))

    print('Rotation finished')
    return img_aft_rot

#-------------------------------------------------------------------------------------------------------------
# function for bi-linear interpolation
def bilinearInterpolation(pixel, img) :

    tran_pixel = np.zeros(3)

    #round the pixels to their nearest smallest integer
    x = math.floor(pixel.item(0))
    y = math.floor(pixel.item(1))

    #get image dimenstions
    [rows, cols, chn] = img.shape
    if (((x > rows - 2) or (x < 0)) or ((y > cols - 2) or y < 0)) :
        return tran_pixel

    alpha = pixel.item(0) - x
    beta = pixel.item(1) - y

   #take weighted average of pixel intensities to get the transformed pixel intensity
    for i in range(3):
        tran_pixel[i] = math.floor(img.item(x, y, i) * (1 - alpha) * (1 - beta)
                                   + img.item(x + 1, y, i) * alpha * (1 - beta)
                                   + img.item(x, y + 1, i) * (1 - alpha) * beta
                                   + img.item(x + 1, y + 1, i) * alpha * beta)

    return  tran_pixel
#-------------------------------------------------------------------------------------------------------------

# function to rotate images using bi-linear interpolation
def rotateImageBI (theta, img, origin):

    #convert angle to degrees
    angle = math.radians(theta)

    #get image dimenstions
    [rows, cols, chn] = img.shape

    #define new image size with padding
    img_aft_rot = np.zeros((2 * max(rows, cols), 2 * max(rows, cols), 3))
    print('New image size: ', img_aft_rot.shape)

    [new_rows, new_cols, new_chn] = img_aft_rot.shape

    # define rotation matrix
    R = np.array([[math.cos(angle), -1 * math.sin(angle)], [math.sin(angle), math.cos(angle)]])

    # find inverse of rotation matrix
    Rt = np.transpose(R)

    for x in range(new_rows - origin.item(0)) :
        for y in range(new_cols - origin.item(1)) :

            #find corresponding pixel coordinates of rotated image in original image
            cord_aft_rot = np.array([[x],[y]])
            cord_bfr_rot = np.matmul(Rt, cord_aft_rot)

            #normalize coordinates for interpolation
            cord_bfr_rot_norm = np.array([[cord_bfr_rot.item(0) - origin.item(0)], [cord_bfr_rot.item(1) - origin.item(1)]])
            #find image intensities using bilinear interpolation
            pixel_intensity = bilinearInterpolation(cord_bfr_rot_norm, img)

            for i in range(chn) :
                img_aft_rot.itemset((x + origin.item(0), y + origin.item(1), i), pixel_intensity[i])

    print('Rotation with BI finished')
    return img_aft_rot

#-------------------------------------------------------------------------------------------------------------
# function to perform affine transformation
def affine (A, img, origin):

    #get image dimenstions
    [rows, cols, chn] = img.shape

    #define new image size with padding
    img_aft_rot = np.zeros((2 * max(rows, cols), 2 * max(rows, cols), 3))
    print('New Image Size: ', img_aft_rot.shape)

    # |x'|      |a00 a01 a02 | |x|
    # |  |  =   |            | |y|  #Affine
    # |y'|      |a10 a11 a12 | |1|

    for x in range(rows) :
        for y in range(cols) :

            #find new pixel coordinates corresponding to rotation
            cord_bfr_trans = np.array([[x + origin.item(0)], [y + origin.item(0)], [1]])
            cord_aft_trans = np.matmul(A, cord_bfr_trans)

            #round-off new pixel indices
            new_index = [math.ceil(cord_aft_trans.item(0) + origin.item(0)), math.ceil(cord_aft_trans.item(1) + origin.item(1))]

            for i in range(chn) :
                img_aft_rot.itemset((new_index[0], new_index[1], i), img.item(x, y, i))

    print('Affine transformation finished')
    return img_aft_rot

#-------------------------------------------------------------------------------------------------------------
# function to perform homography transformation
def homography (H, img, origin):

    #get image dimenstions
    [rows, cols, chn] = img.shape

    #define new image size with padding
    img_aft_homo = np.zeros((2 * max(rows, cols), 2 * max(rows, cols), 3))
    print('New Image Size: ', img_aft_homo.shape)

    # |x"|      |h00 h01 h02 | |x|
    # |y"|  =   |h10 h11 h12 | |y| #Homography => Final pixel coordinates x' = x"/w and y' = y"/w
    # |w |      |h20 h21 h22 | |1|

    for x in range(rows) :
        for y in range(cols) :

            #find new pixel coordinates corresponding to rotation
            cord_bfr_homo = np.array([[x + origin.item(0)], [y + origin.item(1)], [1]])
            cord_aft_homo = np.matmul(H, cord_bfr_homo)

            #round-off new pixel indices
            new_index = [math.ceil(cord_aft_homo.item(0) / cord_aft_homo.item(2) + origin.item(0)),
                         math.ceil(cord_aft_homo.item(1) / cord_aft_homo.item(2) + origin.item(1))]

            for i in range(chn) :
                img_aft_homo.itemset((new_index[0], new_index[1], i), img.item(x, y,i))

    print('Homography finished')
    return img_aft_homo
#------------------------------------------------------------------------------------------------------
