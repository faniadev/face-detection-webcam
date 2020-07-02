import numpy as np
import cv2

lbp_file = '/Users/faniardelia/anaconda3/lib/python3.7/site-packages/cv2/data/lbpcascade_frontalface.xml'
(width, height) = (130, 100)     
face_cascade = cv2.CascadeClassifier(lbp_file)
video = cv2.VideoCapture(0)

while True:
    #img = cv2.imread('150980.jpg')
    ret, img = video.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        face = gray[y:y + h, x:x + w] 
        face_resize = cv2.resize(face, (width, height))
        
    cv2.imshow('webcam_lbp', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.waitKey(0)
cv2.destroyAllWindows()
