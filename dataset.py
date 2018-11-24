import cv2
import numpy as np
import csv
import os
from pathlib import Path
facedetect=cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
samplenum=0
cam=cv2.VideoCapture(0)
fcfile = Path('dataset/facelist.csv')
if not fcfile.is_file():
    fc=open(fcfile,"w+")
    fc.write("ID,Name")
    fc.close()
found=False
while True:
    id=input("Enter ID : ")
    with open('dataset/facelist.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
                if row[0]==str(id):
                    found=True
                    break
    if not found:
        break
name=input("Enter Name : ")
fileo=open(fcfile,"a")
fileo.write("\n")
fileo.write('"'+str(id)+'","'+name+'"')
fileo.close()
while True:
    ret,img=cam.read()
    img=cv2.flip(img,1)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        samplenum=samplenum+1
        pathn="dataset/faces/user."+str(id)+"."+str(samplenum)+".jpg"
        cv2.imwrite(pathn,gray[y:y+h+100,x:x+w+100])
        cv2.rectangle(img,(x-50,y-50),(x+w+50,y+h+50),(258,0,245),5)
        cv2.waitKey(100)
    cv2.imshow("facedetection Window",img)
    cv2.waitKey(1)
    if samplenum>80 :
        break
cam.release()
cv2.destroyAllWindows()
os.system('python trainer.py')


