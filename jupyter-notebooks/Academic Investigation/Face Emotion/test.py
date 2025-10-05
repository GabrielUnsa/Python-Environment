import numpy as np
import cv2
# haarcascade_frontalface_alt.xml se encuentra con archivos 
face_cascade = cv2.CascadeClassifier('/usr/local/lib/python3.6/dist-packages/cv2/data/haarcascade_frontalface_alt.xml')

cap = cv2.VideoCapture(0)
 
while(True):
    # ret valor booleano si la camara esta disponible
    #img pixel capturados
    ret, img = cap.read()
 
    #apply same face detection procedures
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
 
    for (x,y,w,h) in faces:
        #print(faces)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) # cuadrado azul
        cv2.imshow( 'img', img )
    if cv2.waitKey(1) & 0xFF == ord('q'): #presionar q to quit
        break
        
cap.release()
cv2.destroyAllWindows() 
