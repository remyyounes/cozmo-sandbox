import sys
import numpy as np
import cv2 as cv
sys.path.append('../lib/')
import helpers

DEBUG = True
if DEBUG:
    print("Face Recognition")
    print(helpers.infoBanner())

cap = cv.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    helpers.detectInImage(frame)

    # Display the resulting frame
    cv.imshow('frame',frame)

    # Exit on 'q' pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
