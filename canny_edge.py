from __future__ import division
import numpy as np 
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())
 

#Load image
img=cv2.imread(args["image"],0)
width,height=img.shape[:2]

#Gaussian kernel self-construct
deviation=0.52                     #choosing deviation 
gauss=np.zeros((5,5))
for i in range(0,5):
    for  j in range(0,5):
        temp=0
        temp=np.exp(-((np.power(i-2,2)+np.power(j-2,2)))/(2*np.power(deviation,2)))
        gauss[i][j]=temp/(2*np.pi*np.power(deviation,2))

#Create
con_gauss=np.zeros_like(img)     

sobel_x=np.zeros_like(img)        
sobel_x=np.int16(sobel_x)

sobel_y=np.zeros_like(img)        
sobel_y=np.int16(sobel_y)

sobel=np.zeros_like(img)         
grad=np.zeros_like(img)
direc=np.zeros_like(img)

nmax=np.zeros_like(img)          


#Convolutional with gaussian kernel
for x in range(0,width):
    for y in range(0,height):
        sum=0
        for i in range(0,5):
            for j in range(0,5):
                if (x-i+2)<0 or (x-i+2)>=width or (y-j+2)<0 or (y-j+2)>=height:
                    result=0
                else:
                    result=img[x-i+2][y-j+2]*gauss[i][j]
                    sum+=result
        con_gauss[x][y]=int(sum)

#Derivative x_direction
Gx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
Gy=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
Gy=np.flipud(np.fliplr(Gy))
for x in range(0,width):
    for y in range(0,height):
        sum=sum1=0
        for i in range(0,3):
            for j in range(0,3):
                if (x-i+1)<0 or (x-i+1)>=width or (y-j+1)<0 or (y-j+1)>=height:
                    result=result1=0
                else:
                    result=con_gauss[x-i+1][y-j+1]*Gx[i][j]
                    result1=con_gauss[x-i+1][y-j+1]*Gy[i][j]
                    sum+=result
                    sum1+=result1
        sobel_x[x][y]=sum
        sobel_y[x][y]=sum1
        if sobel_x[x][y]<0:
            sobel_x[x][y]=0
        if sobel_x[x][y]>255:
            sobel_x[x][y]=255
        if sobel_y[x][y]<0:
            sobel_y[x][y]=0
        if sobel_y[x][y]>255:
            sobel_y[x][y]=255
        grad[x][y]=np.sqrt(np.power(sobel_x[x][y],2)+np.power(sobel_y[x][y],2))

sobel_y=np.uint8(sobel_y)
sobel_x=np.uint8(sobel_x)



#Intensity of gradient




#Direction of gradient
direc=np.arctan2(sobel_y,sobel_x)*180/np.pi

#Non-max suppresion
for i in range(nmax.shape[0]):
    for j in range(nmax.shape[1]):
        if direc[i][j]<0:
            direc[i][j]+=360
        if ((j+1)<nmax.shape[1]) and((j-1)>=0) and ((i+1)<nmax.shape[0]) and ((i-1)>=0):
            #0 degrees
            if (direc[i][j]>=337.5) or (direc[i][j]<22.5) or (direc[i][j]>=157.5 and direc[i][j]<202.5):
                if grad[i][j]>=grad[i][j-1] and grad[i][j]>=grad[i][j+1]:
                    nmax[i][j]=grad[i][j]
                else:
                    nmax[i][j]=0
            #45 degrees
            if (direc[i][j]>=22.5 and direc[i][j]<67.5) or (direc[i][j]>=202.5 and direc[i][j]<247.5):
                if grad[i][j]>=grad[i-1][j+1] and grad[i][j]>=grad[i+1][j-1]:
                    nmax[i][j]=grad[i][j]
                else:
                    nmax[i][j]=0
            #90 degrees
            if (direc[i][j]>=67.5 and direc[i][j]<112.5) or (direc[i][j]>=247.5 and direc[i][j]<292.5):
                if grad[i][j]>=grad[i-1][j] and grad[i][j]>=grad[i+1][j]:
                    nmax[i][j]=grad[i][j]
                else:
                    nmax[i][j]=0
            #135 degrees
            if (direc[i][j]>=112.5 and direc[i][j]<157.5) or (direc[i][j]>=292.5 and direc[i][j]<337.5):
                if grad[i][j]>=grad[i-1][j-1] and grad[i][j]>=grad[i+1][j+1]:
                    nmax[i][j]=grad[i][j]
                else:
                    nmax[j][j]=0

'''Double thresholding
Use Otsu's method to find low and high thresholding:high threshold=otsu'threshold;low threshold=0.5*high threshold
'''
#Otsu thresholding
pixel_number=width*height
final_variance=100000
threshold_value=-1
final_between=0
hist,bins=np.histogram(nmax.ravel(),256,[0,256])
new_bins=np.delete(bins,-1)
for i in range(0,255):
    weight_b=0
    weight_f=0
    total_pixelsb=np.sum([hist[:i]])
    if total_pixelsb==0:
        continue
    total_pixelsf=np.sum([hist[i:]])
    if total_pixelsf==0:
        break
    
    weight_b+=np.sum(hist[:i])/pixel_number
    weight_f=1-weight_b
    
    mean_b=np.sum(hist[:i]*new_bins[:i])/np.sum(hist[:i])
    var_b=np.sum(((new_bins[:i]-mean_b)**2)*new_bins[:i])/np.sum(hist[:i])
    
    mean_f=np.sum(hist[i:]*new_bins[i:])/np.sum(hist[i:])
    var_f=np.sum(((new_bins[i:]-mean_f)**2)*new_bins[i:])/np.sum(hist[i:])

    # between class variance 
    between_variance=weight_b*weight_f*((mean_b-mean_f)**2)
    if between_variance>final_between:  
        final_between=between_variance
        threshold_value=i
print (threshold_value)

   #Find strong edges and weak edges
high=threshold_value
low=high/2
strongs=[]
thresh=np.zeros_like(img)
for i in range(nmax.shape[0]):
    for j in range(nmax.shape[1]):
        if nmax[i][j]>high:
            thresh[i][j]=1

        elif nmax[i][j]<high and nmax[i][j]>low:
            thresh[i][j]=0.5
            strongs.append((i,j))

        else:
            nmax[i][j]=0

#Edge tracking by hysteresis
dx=[-1,-1,-1,0,1,1,1,0]
dy=[-1,0,1,1,1,0,-1,-1]
value=[]
for i in range(len(strongs)):
    x,y=strongs[i]
    new=0
    for j in range(8):
        if (x+j)>0 and (y+j)>0 and (x+j)<nmax.shape[0] and (y+j)<nmax.shape[1]:
            if thresh[x+j][y+j]==1:
                break
            else:
                new+=1
    if new==8:
        nmax[x][y]=0
cv2.imshow('sample',nmax)
cv2.imwrite('result5.jpg',nmax)
cv2.waitKey(0)
cv2.destroyAllWindows()