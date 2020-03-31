import cv2
import numpy as np
img=cv2.imread('C:\\Users\\chaki\\Desktop\\imageproccesing\\road.jpg',0)

def click(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print(x,",",y)
        font=cv2.FONT_HERSHEY_SIMPLEX
        strXY=str(x)+" "+str(y)
        cv2.putText(img,strXY,(x,y),font,1,(255,255,0),2)
        cv2.imshow("image",img)
x=len(img)
y=len(img[0])
xi=x//90
yi=y//90
matrix=[[0 for _ in range(yi)] for _ in range(xi)]
print(xi,yi,x,y)
print("hi",img[339][581])
for i in range(0,xi-1):
    for j in range(0,yi-1):
        ans=0
        print("---",(i+1)*90,(j+1)*90)
        for tempi in range(0,90):
            for tempj in range(0,90):
                ans+=img[(i*90)+tempi][(j*90)+tempj]
        print(i,j)
        matrix[i][j]=ans
    print()
print(matrix)
cv2.destroyAllWindows()
cv2.imshow("image",img)
cv2.setMouseCallback("image",click)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(img)
