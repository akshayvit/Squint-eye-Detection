from __future__ import division
import cv2
import numpy as np
import dlib
import pyautogui
from math import *
import os
import signal

font = cv2.FONT_HERSHEY_SIMPLEX 
  
# org 
org = (50, 50) 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2
la,lb=0.0,0.0

cap = cv2.VideoCapture(0)
originx,originy=pyautogui.position().x,pyautogui.position().y

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\\Users\\user1\\Desktop\\hitum\\facial-landmarks\\shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier(r'D:\\python3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'D:\\python3\\Lib\\site-packages\\cv2\\data\\haarcascade_righteye_2splits.xml')

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def distance(p1,p2):
    return sqrt((p1.x-p2[0])*(p1.x-p2[0])+(p1.y-p2[1])*(p1.y-p2[1]))

def clicker(x,y,z,w):
    global la,lb
    a=sqrt((x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1]))
    b=sqrt((z[0]-w[0])*(z[0]-w[0])+(z[1]-w[1])*(z[1]-w[1]))
    print(a,b,la,lb,abs(la-a),abs(lb-b))
    if(lb!=0.0 and abs(lb-b)>=9.6):
        pyautogui.click()
    if(la!=0.0 and abs(la-a)>=2.7):
        os.kill(os.getpid(),signal.SIGTERM)
    la=a
    lb=b
while True:
    _, frame = cap.read()
    img=frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray)
    for (ex,ey,ew,eh) in eyes:
        
        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        '''if firstx!=-1:
            pyautogui.moveRel(firstx*0.01, firsty*0.01, duration = 0.05)
            firstx,firsty=firstx+ex*0.01,firsty+ey*0.01
        else:
            firstx,firsty=ex*0.01,ey*0.01
            '''
        roi_gray2 = gray[ey:ey+eh, ex:ex+ew]
        roi_color2 = img[ey:ey+eh, ex:ex+ew]
        circles = cv2.HoughCircles(roi_gray2,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
        try:
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(roi_color2,(i[0],i[1]),i[2],(255,255,255),2)
                print("drawing circle at %f,%f",i[0],i[1])
                # draw the center of the circle
                cv2.circle(roi_color2,(i[0],i[1]),2,(255,255,255),3)
                #pyautogui.moveTo(i[0], i[1], 2)
                originx=originx+i[0]
                originy=originy+i[1]
                faces = detector(gray)
                for face in faces:
                    landmarks = predictor(gray, face)
                    mp=landmarks.part(27)
                    cr = midpoint(landmarks.part(37), landmarks.part(40))
                    cl = midpoint(landmarks.part(44), landmarks.part(47))
                    cv2.putText(frame, "SQUINT EYE - "+str(((distance(mp,cr)-distance(mp,cl))*100)/min((distance(mp,cr),distance(mp,cl))))+"%", org, font,fontScale, color, thickness, cv2.LINE_AA)
                    #clicker(right_point,left_point,center_top,center_bottom)
        except Exception as e:
            pass
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
