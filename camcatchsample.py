# -*- coding: utf-8 -*-
"""
Created on Mon Nov 03 12:18:02 2014

@author: Acer
"""
import random
import cv2
cv2.namedWindow('len',0)
capture=cv2.VideoCapture(0)
p1x=0
p1y=0
p2x=639
p2y=479
k=1
def p1xchange(a):
    global p1x
    global p1y
    global p2x
    global p2y
    p1x=a
    #p1y=p2y-(p2x-p1x)
'''def p2xchange(a):
    global p1x
    global p1y
    global p2x
    global p2y
    p2x=a
    p2y=p1y+(p2x-p1x)'''
def p1ychange(a):
    global p1x
    global p1y
    global p2x
    global p2y
    p1y=a
    #p1x=p2x-(p2y-p1y)
'''def p2ychange(a):
    global p1x
    global p1y
    global p2x
    global p2y
    p2y=a
    p2x=p1x+(p2y-p1y)'''
def kchange(a):
    global k
    k=a
cv2.createTrackbar('k','len',k,8,kchange)
cv2.createTrackbar('p1x','len',p1x,639,p1xchange)
cv2.createTrackbar('p1y','len',p1y,479,p1ychange)
#cv2.createTrackbar('p2x','len',p2x,639,p2xchange)
#cv2.createTrackbar('p2y','len',p2y,479,p2ychange)
while True:
    a,b=capture.read()
    c=cv2.cvtColor(b,7)
    #c=cv2.equalizeHist(c)
    cv2.imshow('len',c[p1y:p1y+150+k*20,p1x:p1x+150+k*20])
    print p1y,p2y,p1x,p2x
    print c[p1y:p1y+150+k*20,p1x:p1x+150+k*20]
    key=cv2.waitKey(1)
    if key==27:
        break
output=cv2.resize(c[p1y:p1y+150+k*20,p1x:p1x+150+k*20],(20,20))
cv2.imwrite('C:\\deeplearning\\facecollect\\face%f.jpg'%random.random(),output)
capture.release()
cv2.destroyAllWindows()