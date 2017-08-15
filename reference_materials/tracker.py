'''
Test program from https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
Source: https://github.com/shantnu/Webcam-Face-Detect
'''

import cv2

cascPath = './haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

vc = cv2.VideoCapture(0)

if vc.isOpened(): #try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while True:
    if rval == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30,30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

#        print( len(faces))
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture
vc.release()
cv2.destroyAllWindows()

