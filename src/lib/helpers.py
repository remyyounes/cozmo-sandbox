import numpy as np
import cv2 as cv
import handy as H
import image_processing as IP

# CASCADE CLASSIFIERS
face_cascade = cv.CascadeClassifier('/usr/local/opt/opencv@3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('/usr/local/opt/opencv@3/share/OpenCV/haarcascades/haarcascade_eye.xml')

# COLORS
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
PURPLE = (255, 0, 255)

# DRAWING UTILS
def drawBox(img, box, color, stroke = 2):
    (y, x, h, w) = box
    cv.rectangle(img, (x, y), (x + w, y + h), color ,stroke)

def drawBoxes(img, color, boxes):
    for box in boxes:
        drawBox(img, box, color)

# IMG PROCESSING UTILS
def crop (img, box):
    (y, x, h, w) = box
    return img[y:y+h, x:x+w]

def copyTo (img, box, content):
    (y, x, h, w) = box
    img[y:y+h, x:x+w] = content

def grayScale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# DELETECTION
def detectFaces(img):
    return face_cascade.detectMultiScale(img, 1.3, 5)

def detectEyes(img):
    return eye_cascade.detectMultiScale(img)

def blur(img, radius):
    return cv.medianBlur(img, radius)

def gaussBlur(img, radius):
    return cv.GaussianBlur(img, (radius, radius), 0)

def toGray(img):
    return cv.cvtColor(im, cv.COLOR_BGR2GRAY)

def captureHistogram():
    return IP.captureHistogram()

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


def detectFingerTips(img, hist):
    ret, frame, contours, defects = H.detectHand(
        img,
        hist,
        sketchContours = True,
        computeDefects = True
    )
    fingerTips = H.extractFingertips(defects, contours, 50, right = True)
    # H.plot(frame, fingerTips, 10, PURPLE)
    return frame
