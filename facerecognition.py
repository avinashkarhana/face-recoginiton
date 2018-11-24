import cv2
import numpy as np
import csv
facedetect=cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
cam=cv2.VideoCapture(0)
recog = cv2.face.LBPHFaceRecognizer_create()
recog.read('recog/traingdata.yml')
id=0
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255,255,255)
lineType = 2
id_name="Unknown"
while True:
    ret,img=cam.read()
    img=cv2.flip(img,1)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(258,0,245),5)
        id,conf=recog.predict(gray[y:y+h,x:x+w])
        print(id)
        if id=='':id=0
        with open('dataset/facelist.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                if row[0]==str(id) and conf<100:id_name=str(row[1])+"  #"+str(conf)
                else:id_name="Unknown"
        cv2.putText(img,str(id_name),(x,y+h),font,fontScale,fontColor,lineType)
    cv2.imshow("face",img)
    if cv2.waitKey(1)==ord('q') :
        break
cam.release()
cv2.destroyAllWindows()
