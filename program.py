import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
from urllib.error import HTTPError
import numpy as np
from cvlib.object_detection import draw_bbox
import time
import pyttsx3
import cv2
import os
from PIL import Image, ImageTk
import pandas as pd
from pathlib import Path
import json
import pytesseract
import speech_recognition as sr


class AiRIS:
    # url = 'http://172.168.8.108/hi.jpg'
    # flashOnURL = 'http://172.168.8.108/flash.on'
    # flashOffURL = 'http://172.168.8.108/flash.off'
    url = 'http://192.168.93.131/hi.jpg'
    flashOnURL = 'http://192.168.93.131/flash.on'
    flashOffURL = 'http://192.168.93.131/flash.off'

    # url = 'http://192.168.0.104/hi.jpg'
    # flashOnURL = 'http://192.168.0.104/flash.on'
    # flashOffURL = 'http://192.168.0.104/flash.off'

    	
    im = None
    harcascadePath = "data/frontface.xml"
    engine = pyttsx3.init()
    r = sr.Recognizer()

    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 160)

    def Union(self, lst1, lst2):
        final_list = list(set(lst1) | set(lst2))
        return final_list

    # Method to check if image is darks
    def img_estim(self, img, thrshld):
        is_light = np.mean(img) > thrshld
        return 'light' if is_light else 'dark'

    # Method to speak
    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    # Method to capture image from device
    def get_imgs(self):
        img_resp = urllib.request.urlopen(self.url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgnp, -1)
        return img

    # Method for object detection
    def objectDetection(self):
        self.speak("Detecting objects")
        timeStart = round(time.time())
        cv2.namedWindow("Object Detection", cv2.WINDOW_AUTOSIZE)
        detected_objects = []
        img = self.get_imgs()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgType = self.img_estim(gray, 125)
        if imgType == 'dark':
            urllib.request.urlopen(self.flashOnURL)

        while True:
            im = self.get_imgs()
            bbox, label, conf = cv.detect_common_objects(im)
            im = draw_bbox(im, bbox, label, conf)
            if len(label) > 0:
                detected_objects = self.Union(label, detected_objects)

            cv2.imshow('Object Detection', im)



            key = cv2.waitKey(5)
            if key == ord('q'):
                break
            timeNow = round(time.time())
            if timeNow - timeStart >= 5:
                break
        urllib.request.urlopen(self.flashOffURL)
        if len(detected_objects) == 1:
            self.speak("Detected " + detected_objects[0])
        elif len(detected_objects) > 1:
            text = "Detected "
            for object in detected_objects:
                if object == detected_objects[-1]:
                    text += " and "
                text += object + ", "
            self.speak(text)
        cv2.destroyAllWindows()

    # Method for text detection
    def textDetection(self):
        self.speak("Reading Text")
        timeStart = round(time.time())
        cv2.namedWindow("Text recongition", cv2.WINDOW_AUTOSIZE)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        img = self.get_imgs()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgType = self.img_estim(gray, 125)
        if imgType == 'dark':
            urllib.request.urlopen(self.flashOnURL)

        while True:

            im = self.get_imgs()
            text = pytesseract.image_to_string(im)
            if(len(text.strip()) > 0):
                self.speak("Reading Text")
                self.speak(text)
            cv2.imshow('Text recongition', im)

            key = cv2.waitKey(5)
            if key == ord('q'):
                break
            timeNow = round(time.time())
            if timeNow - timeStart >= 10:
                break
        urllib.request.urlopen(self.flashOffURL)
        cv2.destroyAllWindows()

    # Method to register new face
    def registerUser(self):
        name = ""
        while True:
            try:
                with sr.Microphone() as mic_source:
                    airis.r.adjust_for_ambient_noise(mic_source, duration=0.5)
                    airis.speak("What is the name of the person?")
                    audio2 = self.r.listen(mic_source, phrase_time_limit=5)
                    input = self.r.recognize_google(audio2)
                    input = input.strip().lower()
                    if input != "cancel":
                        name = input
                        break
                    else:
                        break
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))
            except sr.UnknownValueError:
                airis.speak("Sorry I couldn't hear you")
        if name != "":
            detector = cv2.CascadeClassifier(self.harcascadePath)
            sampleNum = 0
            tm = str(round(time.time()))
            img = self.get_imgs()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgType = self.img_estim(gray, 125)
            if imgType == 'dark':
                urllib.request.urlopen(self.flashOnURL)
            while(True):
                img = self.get_imgs()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)

                # For creating a rectangle around the image
                for (x, y, w, h) in faces:
                    # Specifying the coordinates of the image as well
                    # as color and thickness of the rectangle.
                    # incrementing sample number for each image
                    cv2.rectangle(img, (x, y), (
                        x + w, y + h), (255, 0, 0), 2)
                    sampleNum = sampleNum + 1
                    # saving the captured face in the dataset folder
                    # TrainingImage as the image needs to be trained
                    # are saved in this folder
                    cv2.imwrite(
                        "TrainingImage/ "+name + "."+tm + '.' + str(
                            sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                    # display the frame that has been captured
                    # and drawn rectangle around it.
                    cv2.imshow('frame', img)
                # wait for 100 milliseconds
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                # break if the sample number is more than 60
                elif sampleNum > 60:
                    break
            # releasing the resources
            # cam.release()
            # closing all the windows
            urllib.request.urlopen(self.flashOffURL)
            cv2.destroyAllWindows()
            with open('data.json') as json_file:
                data = json.load(json_file)
                data[tm] = name
            with open('data.json', 'w') as outfile:
                outfile.write(json.dumps(data))
            json.dumps(data)
            self.trainImages()
        else:
            self.speak("Operation cancelled")
    # Training the images saved in training image folder

    def trainImages(self):
        # Local Binary Pattern Histogram is an Face Recognizer
        # algorithm inside OpenCV module used for training the image dataset
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        # creating detector for faces
        detector = cv2.CascadeClassifier(self.harcascadePath)
        # Saving the detected faces in variables
        faces, Id = self.getImagesAndLabels("TrainingImage")
        # Saving the trained faces and their respective ID's
        # in a model named as "trainner.yml".
        # print(Id)
        recognizer.train(faces, np.array(Id))
        recognizer.save("TrainingImageLabel/Trainner.yml")

        # Displaying the message
        self.speak("Face has been registered")

    def getImagesAndLabels(self, path):
        # get the path of all the files in the folder
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faces = []
        # creating empty ID list
        Ids = []
        # now looping through all the image paths and loading the
        # Ids and the images saved in the folder
        for imagePath in imagePaths:
            # loading the image and converting it to gray scale
            pilImage = Image.open(imagePath).convert('L')
            # Now we are converting the PIL image into numpy array
            imageNp = np.array(pilImage, 'uint8')
            # getting the Id from the image
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            # extract the face from the training image sample
            faces.append(imageNp)
            Ids.append(Id)
        return faces, Ids

    # Detect face

    def detectFace(self):
        self.speak("Identifying person")
        timeStart = round(time.time())
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("TrainingImageLabel/Trainner.yml")

        faceCascade = cv2.CascadeClassifier(self.harcascadePath)
        detected_face = []
        timeStart = round(time.time())
        img = self.get_imgs()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgType = self.img_estim(gray, 125)
        if imgType == 'dark':
            urllib.request.urlopen(self.flashOnURL)
        while(True):
            img = self.get_imgs()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)
            for(x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)
                Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                if(conf < 60):
                    if int(Id) not in detected_face:
                        if "Unknown" in detected_face:
                            detected_face[detected_face.index(
                                "Unknown")] = int(Id)
                        else:
                            detected_face.append(int(Id))
                else:
                    if "Unknown" not in detected_face and len(detected_face) == 0:
                        detected_face.append("Unknown")
            cv2.imshow('im', img)
            if (cv2.waitKey(1) == ord('q')):
                break
            timeNow = round(time.time())
            if timeNow - timeStart >= 3:
                break
        if len(detected_face) >= 1:
            all_faces = []
            with open('data.json') as json_file:
                all_faces = json.load(json_file)
            if len(detected_face) == 1:
                for face in detected_face:
                    if(face == "Unknown"):
                        self.speak("Unidentified person detected")
                        cv2.destroyAllWindows()
                        urllib.request.urlopen(self.flashOffURL)
                        self.registerUser()
                    else:
                        self.speak("Detected " + all_faces[str(face)])
            elif len(detected_face) > 1:
                text = "Detected "
                for face in detected_face:
                    text += all_faces[str(face)] + ", "
                self.speak(text)
        else:
            self.speak("Unkown person detected")
            cv2.destroyAllWindows()
            urllib.request.urlopen(self.flashOffURL)
            self.registerUser()
        urllib.request.urlopen(self.flashOffURL)
        cv2.destroyAllWindows()

    # Method for live object detection
    def liveDetection(self):
        self.speak("Detecting live objects")
        cv2.namedWindow("Object Detection", cv2.WINDOW_AUTOSIZE)
        detected_objects = []
        img = self.get_imgs()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgType = self.img_estim(gray, 125)
        if imgType == 'dark':
            urllib.request.urlopen(self.flashOnURL)

        while True:
            im = self.get_imgs()
            bbox, label, conf = cv.detect_common_objects(im)
            im = draw_bbox(im, bbox, label, conf)
            if len(label) > 0:
                for detected_object in label:
                    self.speak(detected_object)

            cv2.imshow('Object Detection', im)

            key = cv2.waitKey(5)
            if key == ord('q'): 
                break
            time.sleep(1)
        urllib.request.urlopen(self.flashOffURL)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    airis = AiRIS()
    while True:
        # print("Select an option\n")
        # print("1. Object Detection\n")
        # print("2. Text Detection\n")
        # print("3. Face Detection\n")
        # print("4. Live Detection\n")
        # print("5. Exit\n")

        try:
            with sr.Microphone() as mic_source:
                airis.r.adjust_for_ambient_noise(mic_source, duration=0.5)
                airis.speak("What would you like me to do?")
                audio2 = airis.r.listen(mic_source, phrase_time_limit=5)
                choice = airis.r.recognize_google(audio2)
                choice = choice.strip().lower()
                # print(choice)
                # choice = int(5)
                if choice in ["object detection", "object", "face recognition", "face", "Kon hai ye", "read text", "text", "live detection", "live", "quit", "exit"]:
                    if choice in ["object detection", "object"]:
                        airis.objectDetection()
                    elif choice in ["read text", "text"]:
                        airis.textDetection()
                    elif choice in ["face recognition", "face", "Kon hai ye"]:
                        airis.detectFace()
                    elif choice in ["live detection", "live"]:
                        airis.liveDetection()
                    else:
                        airis.speak("Goodbye and have a nice day")
                        break
                else:
                    airis.speak("Sorry I cannot understand your request")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        except sr.UnknownValueError:
            airis.speak("Sorry I couldn't hear you")
        except TypeError:
            print("Enter a valid choice\n\n")
        except HTTPError:
            print("Couldn't connect to camera\n\n")
        time.sleep(1)
    quit(0)
