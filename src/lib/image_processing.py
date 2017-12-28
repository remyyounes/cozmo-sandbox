import cv2
import numpy as np
import math
import helpers


alpha = 10;
def update(angle):
    alpha = angle
    # win = 'foo'
    # cv2.namedWindow(win)
    # cv2.createtrackbar('angle', win,  135 , 180, update)

def captureHistogram(source = None):
    source = 0 if source is None else source
    if source is not None and str(type(source)) != "<class 'int'>":
        raise ValueError("source: integer value expected")
    cap = cv2.VideoCapture(source)

    boxSize = 100
    centerX = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    centerY = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 3)
    padding = boxSize + 10
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        box1 = (centerY + padding * 0, centerX + padding * 0, boxSize, boxSize)
        box2 = (centerY + padding * 0, centerX + padding * 1, boxSize, boxSize)
        box3 = (centerY + padding * 1, centerX + padding * 0, boxSize, boxSize)
        box4 = (centerY + padding * 1, centerX + padding * 1, boxSize, boxSize)
        box5 = (centerY + padding * 2, centerX + padding * 0, boxSize, boxSize)
        box6 = (centerY + padding * 2, centerX + padding * 1, boxSize, boxSize)


        helpers.drawBoxes(frame, helpers.PURPLE, [box1, box2, box3, box4])

        boxOne = helpers.crop(frame, box1)
        boxTwo = helpers.crop(frame, box2)
        boxThree = helpers.crop(frame, box3)
        boxFour = helpers.crop(frame, box4)

        boxFourBlurred = helpers.blur(boxFour, 15)

        helpers.copyTo(frame, box5, boxFour)
        helpers.copyTo(frame, box6, boxFourBlurred)


        cv2.imshow("Video Feed", frame)
        key = cv2.waitKey(10)
        if key == 97:
            objectColor = finalHistImage
            cv2.destroyAllWindows()
            break
        if key == 27:
            cv2.destroyAllWindows()
            cap.release()
            exit()

    hsvObjectColor = cv2.cvtColor(objectColor, cv2.COLOR_BGR2HSV)
    objectHist = cv2.calcHist([hsvObjectColor], [0,1], None, [12,15], [0,180,0,256])
    cv2.normalize(objectHist, objectHist, 0,255,cv2.NORM_MINMAX)
    return objectHist
