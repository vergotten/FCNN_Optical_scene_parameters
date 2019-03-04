import os

os.chdir("C:/Users/user/fcnn")

dir_data = "dataset2"
dir_seg = dir_data + "/proper_mask/"; print(dir_seg)
dir_img = dir_data + "/image/"; print(dir_img)

import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
## seaborn has white grid by default so I will get rid of this.
sns.set_style("whitegrid", {'axes.grid' : False})

ldseg = np.array(os.listdir(dir_seg)); #print(ldseg) #папка с annotateted images, list of imgs.png

'''pick the first image file'''
fnm = ldseg[0]; print('first file is ' + str(ldseg[0])) #filename - first image in folder

'''read in the original image and segmentation labels'''
seg = cv2.imread(dir_seg + fnm ); print(seg.shape) # (300, 400, 3) #mask image

print(dir_img + fnm)
img_is = cv2.imread(dir_img + fnm ); print(img_is.shape)              #original image

print("seg.shape={}, img_is.shape={}".format(seg.shape,img_is.shape))

'''Check the number of labels'''
mi, ma = np.min(seg), np.max(seg)
n_classes = ma - mi + 1
print("minimum seg = {}, maximum seg = {}, Total number of segmentation classes = {}".format(mi,ma, n_classes))

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)
ax.imshow(img_is)
ax.set_title("original image")
plt.show()

fig = plt.figure(figsize=(15,10))
for k in range(mi,ma+1):
    ax = fig.add_subplot(3,5,k+1) #rows, columns, index
    ax.imshow((seg == k)*1.0)
    ax.set_title("label = {}".format(k))

plt.show()
