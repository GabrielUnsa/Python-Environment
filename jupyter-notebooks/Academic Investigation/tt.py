import numpy as np
import cv2
# haarcascade_frontalface_alt.xml se encuentra con archivos 
face_cascade = cv2.CascadeClassifier('/usr/local/lib/python3.6/dist-packages/cv2/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)
 
while(True):
    # ret valor booleano si la camara esta disponible
    #img pixel capturados
    ret, img = cap.read()
    ret2, img2 = cap.read()
    ret3, img3 = cap.read()
    #apply same face detection proceduresq
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (x,y,w,h) in faces:
        #print(faces)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) # cuadrado 
        '''center = (x+w//2, y+h//2)
        radius = (w+h)//4
        cv2.circle(img, center, radius, (255, 0, 0), 2)'''
        facedata = img[y:y+h, x:x+w]
        #cv2.imshow( 'facedata', facedata )
    cv2.imshow( 'facedata1.3', img )
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        #print(faces)
        cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2) # cuadrado 
        '''center = (x+w//2, y+h//2)
        radius = (w+h)//4
        cv2.circle(img, center, radius, (255, 0, 0), 2)'''
        facedata = img[y:y+h, x:x+w]
        #cv2.imshow( 'facedata1.1', facedata )
    cv2.imshow( 'facedata1.1', img2 )
    if cv2.waitKey(1) & 0xFF == ord('q'): #presionar q to quit
        break
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        #print(faces)
        cv2.rectangle(img3,(x,y),(x+w,y+h),(0,255,0),2) # cuadrado 
        '''center = (x+w//2, y+h//2)
        radius = (w+h)//4
        cv2.circle(img, center, radius, (255, 0, 0), 2)'''
        facedata = img[y:y+h, x:x+w]
        #cv2.imshow( 'facedata1.1', facedata )
    cv2.imshow( 'facedata1.2', img3 )
    if cv2.waitKey(1) & 0xFF == ord('q'): #presionar q to quit
        break
        
cap.release()
cv2.destroyAllWindows() 