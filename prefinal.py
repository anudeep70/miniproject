import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
img=cv2.imread('C:\\Users\\chaki\\Desktop\\imageproccesing\\imageproces.jpg')
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
for i in fg:
    print(sum(i))
def click(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print(x,",",y)
        font=cv2.FONT_HERSHEY_SIMPLEX
        strXY=str(x)+" "+str(y)
        cv2.putText(img,strXY,(x,y),font,1,(255,255,0),2)
        cv2.imshow("image",img)
x=len(img)
y=len(img[0])
pix=40
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
        img = cv2.putText(img,"1" if ans==58905 else "0",(j*pix+pix//2+1,i*pix+pix//2+1),cv2.FONT_HERSHEY_SIMPLEX,1, (225,225,0),2, cv2.LINE_AA)
        #print(i,j)
        matrix[i][j]=ans
    #print()
#print(matrix)
cv2.destroyAllWindows()
cv2.imshow("image",img)
cv2.setMouseCallback("image",click)
cv2.waitKey(0)
cv2.destroyAllWindows()
#print(img)
