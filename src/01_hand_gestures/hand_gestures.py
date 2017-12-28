import sys
sys.path.append('../lib/')
import numpy as np
import cv2 as cv
import handy as H
import helpers

DEBUG = True

cap = cv.VideoCapture(0)

scene_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
scene_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

if DEBUG:
    print("Hand Gestures Recognition")
    print(helpers.infoBanner())
    print('scene_width', scene_width)
    print('scene_height', scene_height)

hist = helpers.captureHistogram()
while(True):
    # Capture frame-by-frame
    ret, img = cap.read()

    # Our operations on the frame come here
    frame = helpers.detectFingerTips(img, hist)

    # Display the resulting frame
    cv.imshow('frame',frame)

    # Exit on 'q' pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
