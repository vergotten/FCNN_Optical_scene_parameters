## FCNN-restore-light-sources
using fully-convolutional neural network in tasks of restoring optical properties 

![GitHub Logo](images/ar.png)

Due to the rapid development of virtual and augmented reality systems the solu-tion of the problem of formation of the natural illumination conditions for virtual world objects in the real environment becomes more relevant. To recover a light sources position and their optical parameters authors propose to use the fully-convolutional neural network (FCNN), which allows catching the 'behavior of light' features. The output of the FCNN is a segmented image with luminance lev-els. As an encoder it was taken the architecture of VGG-16 with layers that pools and convolves an input and wisely classifies it to one of a class which characterizes its luminance. The image dataset was synthesized with use of the physically correct photorealistic rendering software. Dataset consists of HDR images that were rendered and then presented as image in color contours, where each color corresponds to the luminance level. Designed FCNN decision can be used in tasks of definition of illuminated areas of a room, restoring illumination parameters, analyzing its secondary illumination and their classification to one of a luminance level, which nowadays is one of a major task in designing of mixed reality systems to place a synthesized object to the real environment and match the speci-fied optical parameters and lighting of a room. 

## Achieved results:
![GitHub Logo](images/res.png)

The full paper via link http://ceur-ws.org/Vol-2344/short1.pdf
