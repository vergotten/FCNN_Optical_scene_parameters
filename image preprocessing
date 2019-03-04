'''#script to change filenames you want'''

#annotations_prepped_train/annotated_masks/ - MASKS
#/images_prepped_train/images/ - IMAGES

path = input("Enter a path: ")

import cv2, os
import glob

os.chdir(path)
for index, oldfile in enumerate(glob.glob("*.png"), start=1):
    newfile = 'img{:03}.png'.format(index)
    os.rename (oldfile,newfile)
    
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def bgr_hsv(bgr):
    bgr_value = np.uint8([[bgr]])
    hsv_value = cv2.cvtColor(bgr_value, cv2.COLOR_BGR2HSV)
    lowerLimit = (hsv_value[0][0][0]-20,100,100)
    upperLimit = (hsv_value[0][0][0]+20,255,255)
    Low = np.array(lowerLimit)
    Up = np.array(upperLimit)
    return(Low, Up)
    
'''script to proper mask images, (1,1,1),(2,2,2)...(6,6,6)'''
import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt

path = 'C:/Users/Vuori/!Python repos/#Convolutional Image classifier from Scratch/dataset2/proper_mask/'

levels = ([255,0,0],
          [170,255,0],
          [2,255,83],
          [0,246,235],
          [0,170,255],
          [0,85,255],
          [0,0,255],np.uint8)

#image[np.where((image==[0,0,0]).all(axis=2))] = [255,255,255]

for i, filename in enumerate(os.listdir(path)):
    os.chdir(path)
    #open image etc
    
    image = cv2.imread(filename)
    
    for j, level in enumerate(levels):
        #do smth with img
        if (j == 7):
            break
        lowerLimit, upperLimit = bgr_hsv(levels[j])
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #plt.show()

        mask = cv2.inRange(hsv,lowerLimit,upperLimit)
        image[mask>0]=(j,j,j)

    cv2.imwrite("img{:03}.png".format(i+1),image)
