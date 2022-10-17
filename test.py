import numpy as np
import os
import pandas as pd
import cv2
from scipy.misc import imread,imresize
import pickle

parent = os.listdir("/home/alex/Desktop/ImageFolder")

x = []
y = []
count = 0
output = 0
#gets the video class..
for vid_class in parent[1:]:
    print(vid_class)
    #get subfolder in each video class
    child = os.listdir("/home/alex/Desktop/ImageFolder" + "/" + vid_class)
    #get each file in the folder
    for class_i in child[1:]:
        sub_child = os.listdir("/home/alex/Desktop/ImageFolder" + "/" + vid_class + "/" + class_i)
        #print(sub_child)
        for image_fol in sub_child[1:]:
           #print(image_fol)
           
           if (vid_class ==  'RACQUETBALL_ACTIVITY' ):
                if(count%4 == 0):
                    image = imread("/home/alex/Desktop/ImageFolder" + "/" + vid_class + "/" + class_i + "/" + image_fol)
                    image = imresize(image , (224,224))
                    x.append(image)
                    y.append(output)
                    #save in video folder
                    cv2.imwrite('/home/alex/Desktop/VideoFolder/' + vid_class + '/' + str(count) + '_' + image_fol,image)
                count+=1
           else:
                if(count%8 == 0):
                    image = imread("/home/alex/Desktop/ImageFolder" + "/" + vid_class + "/" + class_i + "/" + image_fol)
                   # print("/home/alex/Desktop/ImageFolder" + "/" + vid_class + "/" + class_i + "/" + image_fol)
                    image = imresize(image , (224,224))
                    x.append(image)
                    y.append(output)
                    #print(x)
                    cv2.imwrite('/home/alex/Desktop/VideoFolder/' + vid_class + '/' + str(count) + '_' + image_fol,image)
                    print('/home/alex/Desktop/VideoFolder/' + vid_class + '/' + str(count) + '_' + image_fol,image)
                count+=1
    output+=1
x = np.array(x)
y = np.array(y)
print("x",len(x),"y",len(y))

  
    
