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

url = 'http://172.168.8.108/hi.jpg'
flashOffURL = 'http://172.168.8.108/flash.off'
flashOnURL = 'http://172.168.8.108/flash.on'
# url = 'http://192.168.140.131/hi.jpg'
# flashOnURL = 'http://192.168.140.131/flash.on'
# flashOnURL = 'http://192.168.140.131/flash.on'
im = None
f_cas = cv2.CascadeClassifier(
    cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')


def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list


def textDetection():
    engine = pyttsx3.init()
    timeStart = round(time.time())
    # voices = engine.getProperty('voices')
    # print(voices)
    cv2.namedWindow("Text recongition", cv2.WINDOW_AUTOSIZE)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    detected_objects = []
    # urllib.request.urlopen(flashOnURL)
    while True:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)

        text = pytesseract.image_to_string(im)
        # im = draw_bbox(im, bbox, label, conf)
        print(text)
        engine.say("Reading Text")
        engine.say(text)
        engine.runAndWait()
        cv2.imshow('Text recongition', im)

        key = cv2.waitKey(5)
        if key == ord('q'):
            break
        timeNow = round(time.time())
        if timeNow - timeStart >= 10:
            break
    # urllib.request.urlopen(flashOffURL)
    cv2.destroyAllWindows()


def faceRecognition():
    timeStart = round(time.time())
    # while True:
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgnp, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = f_cas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    cv2.namedWindow("Face Detection", cv2.WINDOW_AUTOSIZE)
    for x, y, w, h in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv2.imshow("Face Detection", img)

    key = cv2.waitKey(5)
    if key == ord('q'):
        return
    timeNow = round(time.time())
    if timeNow - timeStart >= 10:
        return

    cv2.destroyAllWindows()


if __name__ == '__main__':
    textDetection()
    # faceRecognition()
