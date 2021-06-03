import numpy
import numpy as np
import cv2

def nothing(x):
    pass
cv2.namedWindow('track')
cv2.createTrackbar('low hue','track',0,255,nothing)
cv2.createTrackbar('low satur','track',0,255,nothing)
cv2.createTrackbar('low value','track',0,255,nothing)
cv2.createTrackbar('upp hue','track',95,255,nothing)
cv2.createTrackbar('upp satur','track',253,255,nothing)
cv2.createTrackbar('upp value','track',255,255,nothing)

check = 0
img = cv2.VideoCapture(0)
while True:
    temp = check
    check = 0
    _, frame = img.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    hsv= cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_hue = cv2.getTrackbarPos('low hue', 'track')
    lower_satur = cv2.getTrackbarPos('low satur', 'track')
    lower_value = cv2.getTrackbarPos('low value', 'track')

    upper_hue = cv2.getTrackbarPos('upp hue', 'track')
    upper_satur = cv2.getTrackbarPos('upp satur', 'track')
    upper_value = cv2.getTrackbarPos('upp value', 'track')


    lower = numpy.array([lower_hue, lower_satur, lower_value])
    upper = numpy.array([upper_hue, upper_satur, upper_value])

    to_show = cv2.inRange(hsv,lower,upper)
    result = cv2.bitwise_and(frame,frame,mask=to_show)
    cv2.imshow('result',result)

    cv2.imshow('image',frame)
    cv2.imshow('result', result)
    k= cv2.waitKey(1) & 0xFF
    if k==27:
        break
    print(check)
cv2.destroyAllWindows()