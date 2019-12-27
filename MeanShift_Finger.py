import numpy as np
import cv2
import math
import time
from ardSerial import *
from Py_commander import *
from matplotlib import pyplot as plt


term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

def getCentroid(contour):
    M = cv2.moments(contour)
    if M['m00']!=0:
            centroid_x = int(M['m10']/M['m00'])
            centroid_y = int(M['m01']/M['m00'])
            return (centroid_x,centroid_y)
    return (0,0)

def getEuclidDist(Point,centroid):
    X_diff=(Point[0]-centroid[0])**2
    Y_diff=(Point[1]-centroid[1])**2
    return ((X_diff+Y_diff)**.5)//1

def getArea(first,sec,third):
    a=getEuclidDist(first,sec)
    if(a<=0):
        print("a ", a)
    b=getEuclidDist(sec,third)
    if(b<=0):
        print("b ", b)

    c=getEuclidDist(first,third)
    if(c<=0):
        print("c ", c)

    s=(a+b+c)/2
    if(s<=0):
        print("s ", s)

    Area=abs(s*(s-a)*(s-b)*(s-c))
    if(Area==0):
        Area=0.0000001

    return (Area)**.5


def test_area(defects,contours):
    Areas=[]
    for i in range(4):
        s,e,f,d = defects[i][0]
        start = tuple(contours[0][s][0])
        end   = tuple(contours[0][e][0])
        far   = tuple(contours[0][f][0])
        Areas.append(getArea(start,end,far))
    for i in range(3):
        if(Areas[i]/Areas[i+1] > 1.5 and Areas[i]/Areas[i+1] < 0.5):
            return False
    return True

def Unit_vec(FingerTip,centroid):
    Vector=(centroid[0]-FingerTip[0],centroid[1]-FingerTip[1])
    Distance=getEuclidDist(FingerTip,centroid)
    Vector=Vector/Distance
    Vector[0]=10*round(Vector[0],1)
    Vector[1]=10*round(Vector[1],1)
    print(Vector)
    return Vector

Divide_factor=2

cap=cv2.VideoCapture(0)
frame = cap.read()[1]
upper_left_coord1=(0,0)
lower_righ_coord1=(frame.shape[0]//Divide_factor,frame.shape[1]//Divide_factor)

ROI1_initial=frame[0:frame.shape[1]//Divide_factor,0:frame.shape[0]//Divide_factor]
ROI1_initial=cv2.GaussianBlur(ROI1_initial,(5,5),0)

x=50
Start=[]
Far=[]
Final_frame=0
Final_centroid=0
while(x>0):
    frame = cap.read()[1]
    ROI1=frame[0:frame.shape[1]//Divide_factor,0:frame.shape[0]//Divide_factor]
    ROI_copy=ROI1.copy()
    ROI1=cv2.GaussianBlur(ROI1,(5,5),0)
    ROI1=cv2.absdiff(ROI1, ROI1_initial)
    ROI1=cv2.cvtColor(ROI1, cv2.COLOR_BGR2GRAY)
    ROI1=cv2.threshold(ROI1, 15, 255, cv2.THRESH_BINARY)[1]
    ROI1 = cv2.dilate(ROI1,(5,5),iterations = 5)
    #ROI1 = cv2.morphologyEx(ROI1, cv2.MORPH_CLOSE, (19,19))
    contours,hierarchy = cv2.findContours(ROI1,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   
    if(len(contours)>0):
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        if((cv2.contourArea(contours[0])>(ROI1.shape[0]*ROI1.shape[1])//8) and cv2.contourArea(contours[0])<(ROI1.shape[0]*ROI1.shape[1])//2):
            hull = cv2.convexHull(contours[0],returnPoints=False)
            defects = cv2.convexityDefects(contours[0],hull)
            centroid=getCentroid(contours[0])
            cv2.circle(ROI_copy,centroid,5,[255,255,255],-1)
            if(centroid[0]!=0 and centroid[1]!=0 and defects is not None): 
                defects=sorted(defects, key=lambda x: x[0][3], reverse = True)
                length=6 if(len(defects)>6) else len(defects)
                if(length>3):
                    defects=defects[0:length]
                    if(length!=4): 
                        defects=sorted(defects, key=lambda x: contours[0][x[0][2]][0][1])
                        defects=defects[0:4]
                    #Get distance between the centroid and the far pts
                    defects=sorted(defects, key=lambda x: getEuclidDist(centroid,contours[0][x[0][2]][0]))
                    Smallest_pt=contours[0] [defects[0] [0][2]] [0]
                    Largest__pt=contours[0] [defects[3] [0][2]] [0]
                    cv2.line(ROI_copy,centroid,(Smallest_pt[0],Smallest_pt[1]),[250,250,250],2)
                    cv2.line(ROI_copy,centroid,(Largest__pt[0],Largest__pt[1]),[250,250,250],2)
                    if(getEuclidDist(centroid,Largest__pt)/getEuclidDist(centroid,Smallest_pt) < 1.5 and test_area(defects,contours)):
                        x-=1
                        defects=sorted(defects, key=lambda x: contours[0][x[0][2]][0][0])
                        Start=[]
                        for i in range(4):
                            s,e,f,d = defects[i][0]
                            start=tuple(contours[0][s][0])
                            end = tuple(contours[0][e][0])
                            far = tuple(contours[0][f][0])
                            if(i==0):#You have 5 fingers u genius
                                cv2.circle(ROI_copy,end,5,[150,150,150],-1)
                                Start.append(end)
                                Far.append
                            Start.append(start)
                            cv2.line(ROI_copy,start,end,[50*(i+1),100*(i+1),20*(i+1)],2)
                            cv2.line(ROI_copy,start,far,[50*(i+1),100*(i+1),20*(i+1)],2)
                            cv2.line(ROI_copy,end,far,[50*(i+1),100*(i+1),20*(i+1)],2)
                            cv2.circle(ROI_copy,start,5,[60,100,50],-1)
                            Final_centroid=centroid
                        cv2.imshow("ROI1",ROI_copy)
    cv2.imshow("After",ROI1)
    cv2.rectangle(frame,upper_left_coord1,lower_righ_coord1,(50,150,200),1)
    cv2.imshow("frame",frame)
    Final_frame=frame
    key=cv2.waitKey(1)
    if(key==ord('s')):
        break

cv2.destroyAllWindows()

#Final_frame = cv2.cvtColor(Final_frame, cv2.COLOR_BGR2HSV)


#Replace this by a function which extracts the ROI for each finger given the finger tip location and the frame
w, h = 15, 15 #Should be proportional to the distance between the left-most and the right most finger tips?..The distance travelled must be proportional as well
Finger_ROI=[]
Finger_hist=[]
Finger_windows=[]
for center in Start:
    #Should a hue ROI be implemented instead?
    #Calcualte unit vector wrt the centroid and adjust the rectangles being drawn with that.
    Direc=Unit_vec(center,centroid)

    Upper_0=int(center[0]-w+Direc[0])
    Upper_1=int(center[1]-h+Direc[1])
    Lower_0=int(center[0]+w+Direc[0])
    Lower_1=int(center[1]+h+Direc[1])

    Finger_ROI.append((Final_frame[Upper_1:Lower_1,Upper_0:Lower_0],(Upper_0,Upper_1)))


def ROI_Hist(roi):
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist

for roi,coor in Finger_ROI:
    Finger_hist.append(ROI_Hist(roi))
    Finger_windows.append((coor[0],coor[1],w,h))

cv2.imshow("Frame", frame)
cv2.imshow("finger0", Finger_ROI[0][0])
cv2.imshow("finger1", Finger_ROI[1][0])
cv2.imshow("finger2", Finger_ROI[2][0])
cv2.imshow("finger3", Finger_ROI[3][0])
cv2.imshow("finger4", Finger_ROI[4][0])

cv2.moveWindow("Frame", 0,100)
cv2.moveWindow("finger0", 650, 0)
cv2.moveWindow("finger1", 650, 150)
cv2.moveWindow("finger2", 650, 300)
cv2.moveWindow("finger3", 650, 450)
cv2.moveWindow("finger4", 650, 600)

#Now we implement the mean-shift
while(True):
    frame = cap.read()[1]
    Original=frame.copy()
    for i in range(5):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],Finger_hist[i],[0,180],1)

        # apply meanshift to get the new location
        ret, Finger_windows[i] = cv2.meanShift(dst, Finger_windows[i], term_crit)

        # Draw it on image
        x,y,w,h = Finger_windows[i]

        #Updating the hist for each finger?
        Finger_ROI[i]=Original[y:y+h,x:x+w].copy()
        Finger_hist[i]=ROI_Hist(Finger_ROI[i])

        cv2.rectangle(frame, (x,y), (x+w,y+h), 255,1)

    cv2.imshow("Frame", frame)
    cv2.imshow("finger0", Finger_ROI[0])
    cv2.imshow("finger1", Finger_ROI[1])
    cv2.imshow("finger2", Finger_ROI[2])
    cv2.imshow("finger3", Finger_ROI[3])
    cv2.imshow("finger4", Finger_ROI[4])

    key=cv2.waitKey(1)
    if(key==ord('s')):
        break