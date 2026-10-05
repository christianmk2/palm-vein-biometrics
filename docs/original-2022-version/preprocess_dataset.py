import os
import cv2
import glob
import imageio
import numpy as np
from scipy import ndimage
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import pyplot

new_set = 'D:/PALM QUEZON/PALM QUEZON DATASET/May 5/raw/'

paths = glob.glob(new_set + "*/*/*.jpg")
processed_path2 = 'D:/PALM QUEZON/PALM QUEZON DATASET/May 5/preprocess/'

users = os.listdir(new_set)
users = users[:len(users)]

ctr = 0
ctr2 = 0
ctr3 = 0

def folder_creation(idx):
    #rename_folder = processed[:-1]+"preprocess1/" #for deleting last slash
    if idx%200 == 0:
        global ctr2
        ctr2 += 1
        l_folder = processed_path2 + "user" + str(f"{ctr2:02d}") + "left"
        r_folder = processed_path2 + "user" + str(f"{ctr2:02d}") + "right"
        isExist = os.path.exists(l_folder)
        if isExist == False:
            os.makedirs(l_folder)
            os.makedirs(r_folder)
        elif isExist == True:
            return

def saving_image(image):
    global ctr
    ctr += 1
    left_path = (processed_path2 + "user" + str(f"{ctr2:02d}") + "left")
    right_path = (processed_path2 + "user" + str(f"{ctr2:02d}") + "right")
    if (ctr ==201):
        ctr = 1
        pyplot.imsave(f"{left_path}/" + str(ctr) + ".jpg", image, cmap='gray')
    elif(ctr<=100):
        pyplot.imsave(f"{left_path}/" + str(ctr) + ".jpg", image, cmap='gray') ##using pyplot to save
    elif(ctr<=200):
        pyplot.imsave(f"{right_path}/" + str(ctr) + ".jpg", image, cmap='gray')

x,y,w,h = 251, 121, 198, 185
for idx, paths in enumerate(paths):
    img_original = cv2.imread(paths, cv2.IMREAD_COLOR)

    # Region of Interest (ROI)
    roi = img_original[y:y+h,x:x+w]

    # Noise Reduction
    noiseReduced = cv2.fastNlMeansDenoising(roi)

    #CLAHE
    kernel = np.ones((7,7),np.uint8)
    img = cv2.morphologyEx(noiseReduced, cv2.MORPH_OPEN, kernel)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    frame = cv2.bitwise_not(img_output)

    #Skeletonize
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame.copy()
    skel = img.copy()
    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    while cv2.countNonZero(img) > 0:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    ret,frame = cv2.threshold(skel, 5, 255, cv2.THRESH_BINARY)

    #x1,y1,w1,h1 = 4, 4, 205, 187
    x1,y1,w1,h1 = 4, 4, 190, 177
    newframe = frame[y1:y1+h1,x1:x1+w1]

    print(idx)
    #folder creation before saving
    folder_creation(idx)
    saving_image(newframe)

users = os.listdir(new_set)
users = users[:len(users)]
print(users)
