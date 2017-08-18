'''
Actual Work
Chhewang Sherpa

MIT License

Copyright (c) 2017 chsherpa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''
import imutils
from imutils.video import VideoStream
import cv2

'''
Hard Coded HaarFacial Cascades
'''
faceHaarCascade ='reference_materials/haarcascade_frontalface_default.xml'
eyeHaarCascarde ='reference_materials/haarcascade_eye.xml'


'''
Detection Method

@param img Image
@param cascade The Haar facial cascade used to detect the box
'''
def detection(img, cascade):
    rects = cascade.detectMultiScale(
        img,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    #Check to return shifted rect for image correction
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

'''
Method to define the box drawn around the face

@:param img The video frame
@:param rects The box
@:param BGRcolor Color Schema for the Box
'''
def drawDetectBox(img, rects, BGRcolor):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1,y1), (x2,y2), BGRcolor, 1, 0, 0)

#MAIN
if __name__ == '__main__':
    faceDetectionSchema = cv2.CascadeClassifier(faceHaarCascade)
    eyeDetectionSchema = cv2.CascadeClassifier(eyeHaarCascarde)

    # Video Capture CV2 method
    # Value zero reflects default video capture
    # First try call imutils library VideoStream: Much more stable than
    # native CV2 library video capture
    try:
        cam = VideoStream().start()
    except:
        deviceNum=0
        cam = cv2.VideoCapture(deviceNum)
        if not(cam.isOpened()):
            print("\n[INFO] Video device could not be opened\n")
            exit(1)

    while True:
        img = cam.read()
        img = imutils.resize(img, width=400)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #gray = cv2.equalizeHist(gray)

        rects = detection(gray, faceDetectionSchema)
        vis = img.copy() #Make a frame copy for reference

        color=(11,134,184) #Goldenrod
        #Color is in BGR color, RGB backwards
        drawDetectBox(img, rects,color)

        # Open Video
        cv2.imshow('Simple Detect', img)

        # Press esc to quit
        if cv2.waitKey(20) == 27:
            break

cv2.destroyAllWindows()







