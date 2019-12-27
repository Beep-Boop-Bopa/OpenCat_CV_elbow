import numpy as np
import cv2
import math
import time
from ardSerial import *
from Py_commander import *
from vision_commander import skin_col_range,getCentroid


def skin_finder2(cap):
        '''
        Puts a box on the image and asks the user to put their skin in it and press r or l depending on which arm to measure from
        Please when you see the image, make sure that the skin color is significantly different from the back fround.
        '''
        while(True):
                ret, frame = cap.read()

                #setting up left rectangle
                upper_left_coord=(3*frame.shape[1]//8-12,frame.shape[0]//4-12)
                lower_righ_coord=(3*frame.shape[1]//8+12,frame.shape[0]//4+12)
                cv2.rectangle(frame,upper_left_coord,lower_righ_coord,(100,150,200),1)

                #setting up right rectangle
                upper_left_coord2=(5*frame.shape[1]//8-12,frame.shape[0]//4-12)
                lower_righ_coord2=(5*frame.shape[1]//8+12,frame.shape[0]//4+12)
                cv2.rectangle(frame,upper_left_coord2,lower_righ_coord2,(200,50,100),1)


                cv2.putText(frame,'Put your skin in the rectangle above and press a',(10,450),cv2.FONT_HERSHEY_SIMPLEX, 0.75,(100,200,100),1,cv2.LINE_AA)
                
                #Making the RGB and HSV frames appear side by side
                HSVimage=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                res = np.hstack((frame,HSVimage))
                cv2.imshow("Image ", res)
                
                key=cv2.waitKey(1)
                if key == ord('a'):
                        return (skin_col_range(HSVimage, (upper_left_coord,lower_righ_coord)),skin_col_range(HSVimage, (upper_left_coord2,lower_righ_coord2)))
                        break

def getAngle2(frame,point):
        difference=(frame.shape[0]//2-point[1])//2
        if difference > 70:
                difference=70
        elif difference < -70:
                difference=-70
        return difference

def AngleSorter(frame,key,center,contour_no):
        Ang=getAngle2(frame,center)
        if key=="r":
                if contour_no==0:
                        return (11,Ang)
                elif contour_no==1:
                        return (8,Ang)
                elif contour_no==2:
                        return (13,Ang)
                elif contour_no==3:
                        return (14,Ang)
        elif key=="l":
                if contour_no==0:
                        return (15,Ang)
                elif contour_no==1:
                        return (12,Ang)
                elif contour_no==2:
                        return (9,Ang)
                elif contour_no==3:
                        return (10,Ang)


def threaded_angle_finder(frame,ranges,key):
        frame=cv2.inRange(frame,ranges[0],ranges[1])
        kernel = np.ones((5,5),np.uint64)
        frame = cv2.dilate(frame,kernel,iterations = 2)#increase the interations to increase the contour areas.

        angle=0
        contours,hierarchy = cv2.findContours(frame,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cv2.line(frame,(0,frame.shape[0]//2),(frame.shape[1],frame.shape[0]//2),(100,0,0),3)
        Angles=[]

        #Sometimes no contours in the image
        if len(contours) > 0:
                cnts = sorted(contours, key = cv2.contourArea, reverse = True)
                #This if statement is in there to make sure that if the user isn't in the frame, then the program doesn't start reading the
                #noise and sending signals
                if cv2.contourArea(cnts[0])>5:
                        #It is assumed that they will have the highest area in the picture
                        number_to_be_analysed=4 if len(cnts)>4 else len(cnts)
                        cnts=cnts[0:number_to_be_analysed]
                        cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])
                        for x in range(number_to_be_analysed):
                            center=getCentroid(cnts[x])
                            cv2.circle(frame,center,10,[200,0,0],-1)
                            cv2.line(frame,center,(center[0],frame.shape[0]//2),(200,0,0),2)
                            Angles.append(AngleSorter(frame,key,center,x))
        return (frame, Angles)

def finger_angles(cap,skin_range_first,skin_range_second,port):
    Angles=[0]*16
    while(True):
        ret, frame = cap.read()

        Org=frame.copy()
        #Gaussian blur
        frame=cv2.GaussianBlur(frame,(5,5),1)
        #Filteration
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#Left middle finger color
        frame2=frame.copy()#right middle finger color

        Frame_angles_l=threaded_angle_finder(frame,skin_range_first,"l")
        Frame_angles_r=threaded_angle_finder(frame2,skin_range_second,"r")

        Combined_frame=cv2.bitwise_or(Frame_angles_l[0],Frame_angles_r[0])
        Combined_frame=cv2.cvtColor(Combined_frame, cv2.COLOR_GRAY2BGR)
        Combined_frame=cv2.bitwise_or(Combined_frame,Org)
        for i in range(len(Frame_angles_l[1])):
                Angles[Frame_angles_l[1][i][0]]=Frame_angles_l[1][i][1]
        for i in range(len(Frame_angles_r[1])):
                Angles[Frame_angles_r[1][i][0]]=Frame_angles_r[1][i][1]
        
        cv2.putText(Combined_frame,str(Angles),(10,450),cv2.FONT_HERSHEY_SIMPLEX, 0.35,(100,200,100),1,cv2.LINE_AA)
        cv2.imshow("Lo and behold!!",Combined_frame)
        wrapper(port,['l',Angles,0.001])
        if cv2.waitKey(1) == ord('a'):
            break


def main():
    port=Port_Opener("usb");
    cap=cv2.VideoCapture(0)
    skin_values=skin_finder2(cap);
    cv2.destroyAllWindows()
    finger_angles(cap,skin_values[0],skin_values[1],port);
    cap.release()
    cv2.destroyAllWindows()

main()