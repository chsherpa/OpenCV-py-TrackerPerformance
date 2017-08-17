'''
Actual Work
Chhewang Sherpa
'''
import imutils
from imutils.video import VideoStream
from imutils import face_utils
import cv2

'''
Hard Coded HaarFacial Cascades
'''
faceHaarCascade ='reference_materials/haarcascade_frontalface_default.xml'
eyeHaarCascarde ='reference_materials/haarcascade_eye.xml'


def detect(img, cascade):
    rects = cascade.detectMultiScale(
        img,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

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

if __name__ == '__main__':
    faceDetectionSchema = cv2.CascadeClassifier(faceHaarCascade)
    eyeDetectionSchema = cv2.CascadeClassifier(eyeHaarCascarde)

    #Video Capture CV2 method
    #Value zero reflects default video capture
    try:
        cam = VideoStream().start()
    except:
        deviceNum=0
        cam = cv2.VideoCapture(deviceNum)
        if not(cam.isOpened()):
            print("\nVideo device could not be opened\n")
            exit(1)

    while True:
        img = cam.read()
        img = imutils.resize(img, width=400)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #gray = cv2.equalizeHist(gray)

        rects = detect(gray,faceDetectionSchema)
        vis = img.copy() #Make a frame copy for reference

        color=(11,134,184)
        #Color is in BGR color, RGB backwards
        drawDetectBox(img, rects,color)

        cv2.imshow('Simple Detect', img)

        if cv2.waitKey(20) == 27:
            break
    cv2.destroyAllWindows()







