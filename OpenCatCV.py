import numpy as np
import cv2
import math
import time
from ardSerial import *
from Py_commander import *


class OpenCatCV:
    def __init__(self, No_Colors=2,Elbow_caridnalitty=0,cap=cv2.VideoCapture(0),Port=Port_Opener("usb")):
        self.No_Colors=No_Colors
        self.Elbow_caridnalitty=Elbow_caridnalitty#Only used for elbow. 0 represent right elbow. 1 represents left elbow.
        self.cap=cap
        self.Port=Port

    def getCentroid(self,contour):
        M = cv2.moments(contour)
        if M['m00']!=0:
                centroid_x = int(M['m10']/M['m00'])
                centroid_y = int(M['m01']/M['m00'])
                return (centroid_x,centroid_y)
        return (0,0)

    def AnglesArray(self, Arr):
        Array=[0]*16
        for i in range(8):
            Array[i*2]=i+8
            Array[i*2+1]=Arr[i]
        return Array

    def getAngle(self,Points,frame):
        '''
        Finds the joint angle.
        Also draws on the frame (2 points and 2 lines)
        '''
        a=Points[1];cv2.circle(frame,a,10,[200,0,0],-1)
        b=Points[0];cv2.circle(frame,b,10,[200,0,0],-1)
        c=(frame.shape[1],b[1]) if (self.Elbow_caridnalitty) else (0,b[1])
        ba = [a[0] - b[0],a[1] - b[1]];cv2.line(frame,a,b,(200,0,0),2)
        bc = [c[0] - b[0],c[1] - b[1]];cv2.line(frame,c,b,(200,0,0),2)
        cv2.rectangle(frame,(frame.shape[1]//10,frame.shape[0]//10),(9*frame.shape[1]//10,9*frame.shape[0]//10),(0,255,0),3)
        divisor=np.linalg.norm(ba) * np.linalg.norm(bc)
        cosine_angle=0
        if divisor!=0:
                cosine_angle = np.dot(ba, bc) / (divisor)
        ang = int(np.degrees(np.arccos(cosine_angle))) - 90
        if ang < -70:
                ang = -70
        elif ang>70:
                ang = 70
        return ang

    def getDistance(self,Point,reference,frame,Distance):
        '''
        1. Measures the difference b/w pts.
        2. Draws circles around the finger tips and horizontal/vertical lines.
        '''
        cv2.circle(frame,Point,10,[200,0,0],-1)
        cv2.line(frame,Point,(Point[0],reference[1]),(200,0,0),2)
        
        difference=3*(reference[1]-Point[1])//4
        difference=(difference//5)*5
#        print(difference)
 #       ratio=Distance/(3*frame.shape[1]//10)
  #      difference//=ratio
   #     print(difference, "\n")
        if difference > 70:
                difference=70
        elif difference < -70:
                difference=-70
        return difference

    def motor_index(self,index):
        if index==0:
            return 7
        elif index==1:
            return 3
        elif index==2:
            return 4
        elif index==3:
            return 0
        elif index==6:
            return 1
        elif index==7:
            return 5
        elif index==8:
            return 2
        elif index==9:
            return 6
    def measureAngles(self,frame,Points=[]):
        '''
        Measures the relevant angles and sends an array back.
        Deals with the graphics visualisation.
        Prints the angles list below. 
        '''
        Angles=[0]*8
        if self.No_Colors==1 and len(Points)==2:
            Angles.fill(self.getAngle(Points),frame)
        elif self.No_Colors==2 and len(Points)==10:
            left_length=Points[4][0]-Points[0][0]
            right_length=Points[9][0]-Points[5][0]
            cv2.circle(frame,Points[4],10,[200,0,0],-1);cv2.line(frame,Points[4],(0,Points[4][1]),(200,250,250),2)
            cv2.circle(frame,Points[5],10,[200,0,0],-1);cv2.line(frame,Points[5],(frame.shape[1],Points[5][1]),(200,250,250),2)
            for i in range(10):
                if i<4:
                    Angles[self.motor_index(i)]=self.getDistance(Points[i],Points[4],frame,left_length)
                elif i>5:
                    Angles[self.motor_index(i)]=self.getDistance(Points[i],Points[5],frame,right_length)
        cv2.putText(frame,str(Angles),(10,450),cv2.FONT_HERSHEY_SIMPLEX, 0.4,(250,0,200),1,cv2.LINE_AA)
        return self.AnglesArray(Angles)

    def colorExtractor(self):
        '''
        Puts a box on the image and asks the user to put their skin in it and press r or l depending on which arm to measure from
        Please when you see the image, make sure that the skin color is significantly different from the back fround.
        '''
        while(True):
            frame = self.cap.read()[1]
            Boxes=[]
            for i in range(self.No_Colors):
                constant=-1 if i%2==0 else 1
                x=4+constant*(self.No_Colors-1)
                upper_left_coord=((x)*frame.shape[1]//8-12,frame.shape[0]//4-12)
                lower_righ_coord=((x)*frame.shape[1]//8+12,frame.shape[0]//4+12)
                cv2.rectangle(frame,upper_left_coord,lower_righ_coord,(50,150,200),1)
                Boxes.append([upper_left_coord,lower_righ_coord])
            keys="anykey" if self.No_Colors==2 else "r or l"
            cv2.putText(frame,'Put the colors in the rectangle and press '+keys,(10,450),cv2.FONT_HERSHEY_SIMPLEX, 0.65,(200,200,200),1,cv2.LINE_AA)

            HSVimage=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            res = np.hstack((frame,HSVimage))
            cv2.imshow("Image ", res)
            
            key=cv2.waitKey(1)
            if (self.No_Colors==1 and (key == ord('r') or key == ord('l'))) or (self.No_Colors==2 and key!=-1):
                #The line below will only be used for the case. If fingers are being used, then it won't play any part
                self.Elbow_caridnalitty=0 if key == ord('r') else 1
                #Creating a list of ranges of color allowed.
                Colors=[]
                for i in range(self.No_Colors):
                    Colors.append(self.col_range_finder(HSVimage,Boxes[i]))
                cv2.destroyAllWindows()
                return Colors

    def col_range_finder(self,frame,coordinates,Color_wavelength_range=6,Saturation_brightness=0.5):
        '''
        Finds the avg. of the pixels in the box and sends back an acceptable range 
        '''
        top=coordinates[0][0]+1
        down=coordinates[1][0]-1
        left=coordinates[0][1]+1
        right=coordinates[1][1]-1

        #Extracting the frame
        myimg=frame[left:right,top:down]

        #Calculating the avg.
        avg_color_per_row = np.average(myimg, axis=0)
        avg=np.average(avg_color_per_row, axis=0)

        #Please change these params below for your skin color and light, environmental conditions
        return [(avg[0]-Color_wavelength_range,int(avg[1]*(1-Saturation_brightness)),10),(avg[0]+Color_wavelength_range,int(avg[1]*(1+Saturation_brightness)),225)]

    def colorFilter(self,colors,framex):
        '''
        Filters the frame for the given color. Then it fits a contour around the areas. Sorts the contour by size. 
        Then:
        1. For elbow, only considers the top <=3 areas. Finds the center, highest point of the leftmost/rightmost contour.
        2. For fingers, only considers the top <=5 areas. Finds the center of the highest 5 contours. Sorts them by columns and returns them.
        '''
        Org=framex.copy()
        framex=cv2.GaussianBlur(framex,(5,5),1)
        framex=cv2.cvtColor(framex, cv2.COLOR_BGR2HSV)#Left middle finger color
        Points=[]
        for ranges in colors:
            frame=framex.copy()
            frame=cv2.inRange(frame,ranges[0],ranges[1])
            kernel = np.ones((5,5),np.uint64)
            frame = cv2.dilate(frame,kernel,iterations = 2)#increase the interations to increase the contour areas.
            contours,hierarchy = cv2.findContours(frame,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            #Sometimes no contours in the image
            if len(contours) > 0:
                cnts = sorted(contours, key = cv2.contourArea, reverse = True)
                if cv2.contourArea(cnts[0])>5:
                    #It is assumed that they will have the highest area in the picture
                    Max_no_contours=5 if self.No_Colors==2 else 3
                    number_to_be_analysed=Max_no_contours if len(cnts)>Max_no_contours else len(cnts)
                    cnts=cnts[0:number_to_be_analysed]
                    for x in range(number_to_be_analysed):
                        Points.append(self.getCentroid(cnts[x]))
                    if len(Points)>0:
                        Points = sorted(Points, key=lambda ctr: ctr[0])    
                        if self.No_Colors==1:
                            Points=[Points[len(Points)-1]] if self.Elbow_caridnalitty else [Points[0]]
                            cnts = sorted(contours, key = cv2.contourArea, reverse = True)
                            cnt=cnts[len(cnts)-1] if self.Elbow_caridnalitty else [cnts[0]]
                            topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
                            Points.append(topmost)
        return Org, Points

def main():
    Cat_player=OpenCatCV()  #OpenCatCV(sys.argv[1]) if len(sys.argv)>1 else OpenCatCV()
    Colors=Cat_player.colorExtractor()
    while True:
        frame, Points=Cat_player.colorFilter(Colors,Cat_player.cap.read()[1])
        time_delay=0.001#sys.argv[2] if len(sys.argv)>2 else 0.001
        wrapper(Cat_player.Port,['i',Cat_player.measureAngles(frame,Points),time_delay])
        cv2.imshow("Stuff ", frame)  
        if cv2.waitKey(1)!=-1:
            cv2.destroyAllWindows()
            break
main()