from __future__ import print_function
import numpy as np 
import cv2
from matplotlib import pyplot as plt
image = cv2.imread('C:\\Users\\chaki\\Desktop\\imageproccesing\\test.jpg')
def printPath(l):
    alpha=0.5
    print("hi from pp")
    global image
    for i,j in l:
        if [i,j]==[-1,-1]:
            break
        i,j=j*90,i*90
        overlay = image.copy()
        output = image.copy()
        cv2.rectangle(overlay, (i,j), (i+90,j+90),(255, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha,0, output)
        image=output
    cv2.imshow("final",image)
outArr=[[-1,-1] for _ in range(10000)] 
img=cv2.imread('C:\\Users\\chaki\\Desktop\\imageproccesing\\test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
cv2.imshow('org', img)  
ret, thresh = cv2.threshold(gray, 0, 255, 
                            cv2.THRESH_BINARY_INV +
                            cv2.THRESH_OTSU) 
kernel = np.ones((3, 3), np.uint8) 
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, 
                            kernel, iterations = 2) 
  
# Background area using Dialation 
bg = cv2.dilate(closing, kernel, iterations = 1) 
  
# Finding foreground area 
dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0) 
ret, fg = cv2.threshold(dist_transform, 0.02
                        * dist_transform.max(), 255, 0)
img=fg
def click(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        #print(x,",",y)
        font=cv2.FONT_HERSHEY_SIMPLEX
        strXY=str(x)+" "+str(y)
        cv2.putText(img,strXY,(x,y),font,1,(255,255,0),2)
        cv2.imshow("image",img)
x=len(img)
y=len(img[0])
pix=90
xi=x//pix
yi=y//pix
color = (255, 0, 0)
thickness = 2
matrix=[[0 for _ in range(yi)] for _ in range(xi)]
#print(xi,yi,x,y)
#print("hi",img[339][581])
for i in range(0,xi):
    for j in range(0,yi):
        ans=0
        img = cv2.rectangle(img,(j*pix,i*pix), ((j+1)*pix,(i+1)*pix), color, thickness) 
        #print("---",(i+1)*pix,(j+1)*pix)
        for tempi in range(0,pix):
            for tempj in range(0,pix):
                ans+=img[(i*pix)+tempi][(j*pix)+tempj]
        img = cv2.putText(img,"1" if ans==135405 else "0",(j*pix+pix//2-7,i*pix+pix//2+7),cv2.FONT_HERSHEY_SIMPLEX,1, (225,225,0),2, cv2.LINE_AA)
        #print(i,j)
        matrix[i][j]=1 if (ans==135405) else 0
    #print()
cv2.destroyAllWindows()
cv2.imshow("image",img)
cv2.setMouseCallback("image",click)
#print(img)
MAX_X = xi-1
MAX_Y = yi-1
count = 0
done = 0
req_x=0
req_y=5
def getGridValue(x,y):
    return matrix[x][y]

def findPath(x,y):
    global count
    global done
    print(x,y)
    if [x,y] in outArr:
        return
    if count >= 500 or done == 1:
        return
    outArr[count][0]=x
    outArr[count][1]=y
    count+=1
    if y < MAX_Y and getGridValue(x,y+1) == 1:
        findPath(x,y+1)
    if x < MAX_X and getGridValue(x+1,y) == 1:
        findPath(x+1,y)
    if y >0 and getGridValue(x,y-1) == 1:
        findPath(x,y-1)
    if x >0 and getGridValue(x-1,y) == 1:
        findPath(x-1,y)
    if x ==req_x and y==req_y:
        done = 1
        return
    if (done == 0):
        count-=1
        outArr[count][0]=-1
        outArr[count][1]=-1
    return
print("hi from fp")
start=list(map(int,input("start point").split()))
req_x,req_y=list(map(int,input("end point").split()))
findPath(start[0],start[1])
print("bye from fp")
printPath(outArr)

