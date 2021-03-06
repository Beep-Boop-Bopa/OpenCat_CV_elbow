#!/usr/bin/python3
import serial
import struct
import sys
import time
import math
import numpy as np

def wrapper(port,task):  #Structure of task is [token, var=[], time]
#    print ('task '+str(task)+'\n')
    if len(task)==2:
        serialWriteByte(port,[task[0]])
    elif isinstance(task[1][0],int):
        serialWriteNumToByte(port,task[0],task[1])
    else:
        serialWriteByte(port,task[1])
 #   print ("task[-1]: "+str(task[-1]))
    time.sleep(task[-1])


def serialWriteNumToByte(port,token,var=[]): # Only to be used for c m u b i l o within Python
    if token == 'l' or token=='i':# or  token=='d':
        var=list(map(lambda x:int(x), var))
        instrStr=struct.pack('b' * len(var), *var)#   print(instrStr)
        last="~"
    elif token =='c' or token =='m' or token =='u' or token =='b':
        instrStr = token + str(var[0])+" "+str(var[1])+'\n'
        last="\n"
    port.write(token.encode())
    port.write(instrStr)
    port.write(last.encode())

def serialWriteByte(port,var=[]):
    token = var[0][0]
    if (token == 'c' or token == 'm' or token=='b' or token=='u') and len(var)>=2:
        instrStr=""
        for element in var:
            instrStr=instrStr +element+" "
    elif token == 'l' or token=='i':#d or token=='d':
        if(len(var[0])>1):
            var.insert(1,var[0][1:])       
        var[1:]=list(map(lambda x:int(x), var[1:]))
        instrStr = struct.pack('b' * len(var[1:]), *var[1:])
    elif token == 'w' or token == 'k':#['k',balance'] 
        if(len(var)==2):
            instrStr= var[0] + var[1]+"\n" #handles the case when there's space between k and skill name
        else:
            instrStr = var[0] + '\n'
    else:
        instrStr = token+'~'

    if token == 'l' or token=='i':        
        port.write(token.encode())
        port.write(instrStr.encode())
        port.write("~".encode())
    else:
        port.write(instrStr.encode())

if __name__ == '__main__':
    counter=0
    port = serial.Serial(port='/dev/ttyUSB0',
                        baudrate=57600,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        bytesize=serial.EIGHTBITS,
                        timeout=1
                        )
    time.sleep(5)
    if len(sys.argv) >= 2:
        if len(sys.argv)==2:
            wrapper(port,[sys.argv[1],10]) #wrapper([token, time])
        else:
            wrapper(port,[sys.argv[1][0],sys.argv[1:],0]) #wrapper([token,['m','2','1'],time])....wrapper(['k',['k','balance'],0])
    else:
        while True:
            for a in np.arange(0, 2 * math.pi, 0.2):
                print (a)
                serialWriteByte(port,["ksit"])
                time.sleep(0.04)
    while True:
        time.sleep(0.01)
        counter=counter+1
        if counter>1000:
            break
        #print("number of chars:" +str(ser.in_waiting))
        if port.in_waiting>0:
            x = port.readline()
            if x != "":
                print (x)