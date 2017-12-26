import numpy as np
import cv2 as cv
import helpers

DEBUG = 1

if DEBUG: print(helpers.infoBanner())

#
cap = cv.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    helpers.detectInImage(frame)

    # Display the resulting frame
    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
#
#

# ====================================
# ====================================
# ====================================






# faceImg = cv.imread('face_detection.jpg')
# helpers.detectInImage(faceImg)
#
# cv.imshow('img',faceImg)
# cv.waitKey(0)
# cv.destroyAllWindows()
