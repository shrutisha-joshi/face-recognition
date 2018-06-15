import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('recognizer/trainningData.yml')
id=0
name=""
font=cv2.FONT_HERSHEY_SIMPLEX
while(1):
    ret,image = cam.read()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);
    faces = faceDetect.detectMultiScale(gray,1.3,5);

    for(x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=rec.predict(gray[y:y+h,x:x+h])
        if (id == 101):
            name="Shrutisha"
        elif (id == 102):
            name="Mumma"
        cv2.putText(image,name,(x,y+h),font,2,(255,0,0),2)
        cv2.imshow("Detect",image)
    if (cv2.waitKey(1)== ord('q')):
        break
cam.release()
cv2.destroyAllWindows()    
