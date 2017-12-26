import numpy as np
import cv2 as cv
import handy as H

# CASCADE CLASSIFIERS
face_cascade = cv.CascadeClassifier('/usr/local/opt/opencv@3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('/usr/local/opt/opencv@3/share/OpenCV/haarcascades/haarcascade_eye.xml')

# COLORS
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0,0,255)
PURPLE = (255,0,255)

# DRAWING UTILS
def drawBox(img, box, color):
    (x, y, w, h) = box
    cv.rectangle(img, (x, y), (x + w, y + h), color ,2)

def drawBoxes(img, color, boxes):
    for box in boxes:
        drawBox(img, box, color)

# IMG PROCESSING UTILS
def crop (img, box):
    (x, y, w, h) = box
    return img[y:y+h, x:x+w]

def grayScale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# DELETECTION
def detectFaces(img):
    return face_cascade.detectMultiScale(img, 1.3, 5)

def detectEyes(img):
    return eye_cascade.detectMultiScale(img)


# MISC UTILS
# TODO: Add relevant msgs
def infoBanner():
    messages = [
        "OpenCV Version " +  cv.__version__
    ]
    return "\n".join(messages)


def detectInImage(img):
    faces = detectFaces(img)
    drawBoxes(img, RED, faces)
    for face in faces:
        imgFace = crop(img, face)
        drawBoxes(imgFace, BLUE, detectEyes(imgFace))


# TODO: move init somewhere else
hist = H.captureHistogram(0)

def detectFingerTips(img):
    ret, frame, contours, defects = H.detectHand(
        img,
        hist, # TODO: remove dangerous global
        sketchContours = True,
        computeDefects = True
    )
    fingerTips = H.extractFingertips(defects, contours, 50, right = True)
    H.plot(frame, fingerTips)
    return frame
