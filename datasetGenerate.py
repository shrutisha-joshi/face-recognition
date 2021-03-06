import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

id=input('enter user id:')
sNum = 0
while(1):
    ret,image = cam.read()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)

    for(x,y,w,h) in faces:
        sNum = sNum+1
        cv2.imwrite("dataSet/User."+str(id)+"."+str(sNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow("Detect",image)
    cv2.waitKey(1)
    if (sNum>20):
        break
cam.release()
cv2.destroyAllWindows()    
