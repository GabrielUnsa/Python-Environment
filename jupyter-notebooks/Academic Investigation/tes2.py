import requests
import urllib as ul
import cv2
import numpy as np 

# haarcascade_frontalface_alt.xml se encuentra con archivos 
face_cascade = cv2.CascadeClassifier('/usr/local/lib/python3.6/dist-packages/cv2/data/haarcascade_frontalface_alt.xml')
url = 'http://192.168.100.3:8080/shot.jpg'
while True:
    imgResp = ul.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp,1)
    print(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),flags = cv2.CV_HAAR_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) # cuadrado azul
        facedata = img[y:y+h, x:x+w]
        cv2.imshow( 'facedata', facedata )
    cv2.imshow( 'img', img )
    if cv2.waitKey(1) & 0xFF == ord('q'): #presionar q to quit
        break
    cv2.waitKey(10)