import cv2
import numpy as np
import os

filter1 = np.array([[0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]], np.uint8)
filter2 = np.ones((3, 3))

list_resize = [2, 3, 5, 7]
list_mosaic = [3, 5, 7, 10]
list_rotation = [45, 135, 225, 315]
list_flip = [0, 1, -1]
list_cvt1 = [0]  
list_cvt2 = [0]  
list_THRESH_BINARY = [50, 100, 150, 200]
list_THRESH_BINARY_INV = [50, 100, 150, 200]
list_THRESH_TRUNC = [50, 100, 150, 200]
list_THRESH_TOZERO = [50, 100, 150, 200]
list_THRESH_TOZERO_INV = [50, 100, 150, 200]
list_gauss = [11, 31, 51, 71]
list_nois_gray = [0]
list_nois_color = [0]
list_dilate = [filter1, filter2]
list_erode = [filter1, filter2]

dataset_path = "dataset_FER_2013/train"
output_path = "CleansingData/train"

if not os.path.exists(output_path):
    os.makedirs(output_path)

num = 0  

def save(emotion, img, prefix):
    global num
    emotion_folder = os.path.join(output_path, emotion)
    if not os.path.exists(emotion_folder):
        os.makedirs(emotion_folder)
    
    filename = f"{prefix}_{num}.jpg"
    cv2.imwrite(os.path.join(emotion_folder, filename), img)
    num += 1

for emotion in os.listdir(dataset_path):
    emotion_path = os.path.join(dataset_path, emotion)
    
    if not os.path.isdir(emotion_path):  
        continue  

    for img_name in os.listdir(emotion_path):
        img_path = os.path.join(emotion_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue  

        for i in list_rotation:
            mat = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), i, 1)
            cnv_img = cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]))
            save(emotion, cnv_img, "rotate")

        for i in list_flip:
            cnv_img = cv2.flip(img, i)
            save(emotion, cnv_img, "flip")

print("Data augmentation completed")
