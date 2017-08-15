'''
Source: https://realpython.com/blog/python/face-detection-in-python-using-a-webcam/
'''

import cv2
import sys
import numpy as np

cascPath = './haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap.open(0)

#Resize Window Size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret==True:
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('Video',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()