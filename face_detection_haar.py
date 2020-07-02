import numpy as np
import cv2

haar_file = 'haarcascade_frontalface_default.xml'
(width, height) = (130, 100)     
face_cascade = cv2.CascadeClassifier(haar_file)
video = cv2.VideoCapture(0)
while True:
    ret, img = video.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        face = gray[y:y + h, x:x + w] 
        face_resize = cv2.resize(face, (width, height))
        
    cv2.imshow('webcam_haar', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.waitKey(0)
cv2.destroyAllWindows()
