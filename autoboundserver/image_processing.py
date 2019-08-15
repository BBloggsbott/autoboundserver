import cv2
import numpy as np
from fastai.vision import open_image, get_transforms
from PIL import Image

def get_approx(cv_img):
    img2gray = cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.float32)/25
    img2gray = cv2.filter2D(img2gray,-1,kernel)
    _,thresh = cv2.threshold(img2gray,250,255,cv2.THRESH_BINARY_INV)
    contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)
    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)
    #perimeter = cv2.arcLength(cnt,True)
    epsilon = 0.01*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    return approx

def get_image_tensor(decoded_image_bytes):
    image_tensor = open_image(decoded_image_bytes)
    image_tensor = image_tensor.apply_tfms(get_transforms()[0], size = 256)
    image_tensor = image_tensor.data.unsqueeze(dim=0)
    image = Image.open(decoded_image_bytes)
    return image, image_tensor

def pil_to_cv(pil_img):
    cv_image = np.array(pil_img)
    cv_image = cv_image[:, :, ::-1].copy()
    return cv_image

