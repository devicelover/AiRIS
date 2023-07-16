import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures
import time
import pyttsx3
import pytesseract
import imageio
 
url=''
flashOnURL=''
flashOffURL=''
im=None
f_cas= cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')

def img_estim(img, thrshld):
    is_light = np.mean(img) > thrshld
    return 'light' if is_light else 'dark'

def faceRecognition():
    timeStart = round(time.time())
    while True:
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        img=cv2.imdecode(imgnp,-1)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        imgType = img_estim(gray, 125)
        print(imgType)
        if imgType == 'dark':
            urllib.request.urlopen(flashOnURL)
        
        key=cv2.waitKey(5)
        if key==ord('q'):
            urllib.request.urlopen(flashOffURL)
            break
        timeNow = round(time.time())
        if timeNow - timeStart >= 10:
            urllib.request.urlopen(flashOffURL)
            break
    cv2.destroyAllWindows()
 
if _name_ == '_main_':
    faceRecognition()
    # faceRecognition()