# importing libraries
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import time
from pathlib import Path
import urllib.request
import json
import pyttsx3

url = 'http://172.168.8.103/hi.jpg'
flashOnURL = 'http://172.168.8.103/flash.on'
flashOffURL = 'http://172.168.8.103/flash.off'
im = None

# The function below is used for checking
# whether the text below is number or not ?


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
# Take Images is a function used for creating
# the sample of the images which is used for
# training the model. It takes 60 Images of
# every new user.


def TakeImages():

    # Both ID and Name is used for recognising the Image
    name = input("Enter name of person: ")
    # cam = cv2.VideoCapture(0)
    # Specifying the path to haarcascade file
    harcascadePath = "data/frontface.xml"
    # Creating the classier based on the haarcascade file.
    detector = cv2.CascadeClassifier(harcascadePath)
    # Initializing the sample number(No. of images) as 0
    sampleNum = 0
    tm = str(round(time.time()))
    urllib.request.urlopen(flashOnURL)
    while(True):
        # Reading the video captures by camera frame by frame
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgnp, -1)
        # Converting the image into grayscale as most of
        # the the processing is done in gray scale format
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # It converts the images in different sizes
        # (decreases by 1.3 times) and 5 specifies the
        # number of times scaling happens
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
    urllib.request.urlopen(flashOffURL)
    cv2.destroyAllWindows()
    with open('data.json') as json_file:
        data = json.load(json_file)
        data[tm] = name
    with open('data.json', 'w') as outfile:
        outfile.write(json.dumps(data))
    json.dumps(data)
    TrainImages()
    # Displaying message for the user
    # res = "Images Saved for ID : " + Id +" Name : "+ name
    # Creating the entry for the user in a csv file
    # row = [Id, name]
    # with open(r'details/details.csv', 'a+') as csvFile:
    # writer = csv.writer(csvFile)
    # Entry of the row in csv file
    # writer.writerow(row)
    # csvFile.close()
    # message.configure(text = res)

# Training the images saved in training image folder


def TrainImages():
    # Local Binary Pattern Histogram is an Face Recognizer
    # algorithm inside OpenCV module used for training the image dataset
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # Specifying the path for HaarCascade file
    harcascadePath = "data/frontface.xml"
    # creating detector for faces
    detector = cv2.CascadeClassifier(harcascadePath)
    # Saving the detected faces in variables
    faces, Id = getImagesAndLabels("TrainingImage")
    # Saving the trained faces and their respective ID's
    # in a model named as "trainner.yml".
    # print(Id)
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    # Displaying the message


def getImagesAndLabels(path):
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


# For testing phase
def TrackImages():

    engine = pyttsx3.init()
    timeStart = round(time.time())
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read("TrainingImageLabel/Trainner.yml")

    harcascadePath = "data/frontface.xml"

    faceCascade = cv2.CascadeClassifier(harcascadePath)
    detected_face = []
    timeStart = round(time.time())
    urllib.request.urlopen(flashOnURL)
    while(True):
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgnp, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for(x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            print(conf)
            if(conf < 60):
                if int(Id) not in detected_face:
                    if "Unknown" in detected_face:
                        detected_face[detected_face.index("Unknown")] = int(Id)
                    else:
                        detected_face.append(int(Id))
            else:
                if "Unknown" not in detected_face and len(detected_face) == 0:
                    detected_face.append("Unknown")
            # if(conf > 75):
            #     noOfFile = len(os.listdir("ImagesUnknown"))+1
            #     cv2.imwrite("ImagesUnknown/Image" +
            #                 str(noOfFile) + ".jpg", im[y:y + h, x:x + w])
            # cv2.putText(im, str(Id), (x, y + h),
            #             font, 1, (255, 255, 255), 2)
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
                    engine.say("Unkown person detected")
                    engine.runAndWait()
                    cv2.destroyAllWindows()
                    urllib.request.urlopen(flashOffURL)
                    TakeImages()
                else:
                    engine.say("Detected " + all_faces[str(face)])
                    engine.runAndWait()
        elif len(detected_face) > 1:
            engine.say("Detected")
            for face in detected_face:
                engine.say(all_faces[str(face)] + ",")
            engine.runAndWait()
    else:
        engine.say("Unkown person detected")
        engine.runAndWait()
        cv2.destroyAllWindows()
        urllib.request.urlopen(flashOffURL)
        TakeImages()
    urllib.request.urlopen(flashOffURL)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # while True:
    TrackImages()
