# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

'''
# local modules
from video import create_capture
from common import clock, draw_str
'''


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

    '''
    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    '''

    #cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    cascPath = './haarcascade_frontalface_default.xml'
    #cascade_fn = cv2.CascadeClassifier(cascPath)
    eyeHaarPath = './haarcascade_eye.xml'
    #nested_fn = cv2.CascadeClassifier(eyeHaarPath)

    #nested_fn  = args.get('--nested-cascade', "../../data/haarcascades/haarcascade_eye.xml")

    cascade = cv2.CascadeClassifier(cascPath)
    nested = cv2.CascadeClassifier(eyeHaarPath)

    #cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')
    cam = cv2.VideoCapture(0)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

    #    t = clock()
        rects = detect(gray, cascade)
        vis = img.copy()
        #Color is in BGR color, RGB backwards
        draw_rects(vis, rects, (11, 134, 184))
        '''
        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                draw_rects(vis_roi, subrects, (0, 255, 0))
                '''
    #    dt = clock() - t

    #    draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv2.imshow('facedetect', vis)

        if cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()