import numpy as np
import os
import cv2

face_cascade= cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")

img=cv2.imread('Avinash.jpg',cv2.IMREAD_COLOR)
cv2.circle(img,(150,150),55,(0,0,255),-1)
cv2.imshow("img",img)

#foracc=cv2.VideoWriter_fourcc(*'MPEG')
cap = cv2.VideoCapture(0)
#out = cv2.VideoWriter("file.mp4", foracc, 10,(1280,720) )
while True:
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x, y, w, h) in faces: 
    	roi_color = frame[y:y+h, x:x+w]
    	color = (255, 0, 0)
    	stroke = 2
    	end_cord_x = x + w
    	end_cord_y = y + h
    	cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    cv2.imshow('frame',frame)
   # out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
#out.release()
cv2.destroyAllWindows()