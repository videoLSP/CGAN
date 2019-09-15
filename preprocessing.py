
# coding: utf-8

# In[ ]:


import numpy as np
import math as mt
import cv2
import matplotlib.pyplot as plt
from keras.applications.imagenet_utils import preprocess_input


# In[ ]:


def resize(img,h,w):
    return cv2.resize(img, (h, w), interpolation=cv2.INTER_AREA)


# In[ ]:


#preprocessing rgb
def rgb_preprocessing(img, width=224, height=224):
    img = central_crop(img, 0.875)
    img= np.expand_dims(img, axis=0)
    img = preprocess_input(img, data_format ='channels_last', mode='caffe')
    img = np.squeeze(img,axis=0)
    img = resize(img,height,width)
    return img


# In[ ]:


def milimeters(mat):
    mat[mat<0]=0
    r1 = np.zeros((mat.shape))
    mat = np.divide(mat,8)
    for i in range(13):
        r1 = r1 + np.power(2,i)*np.mod(mat,2)
        mat = np.divide(mat,2)
    return r1


# In[ ]:


def escale_data(mat):
    minItem = np.float64(mat.min())
    maxItem = np.float64(mat.max())
    if (maxItem - minItem) != 0:
        tmpMat = (mat - minItem) / np.float64(maxItem - minItem) # scale to [0-1]
    return tmpMat


# In[ ]:


def viridis(mat):
    cmap=plt.cm.viridis
    im=cmap(mat)
    return im[:,:,:3]


# In[ ]:


def central_crop(image, central_fraction):
    """Crop the central region of the image.
    Remove the outer parts of an image but retain the central region of the image
    along each dimension. If we specify central_fraction = 0.5, this function
    returns the region marked with "X" in the below diagram.
       --------
      |        |
      |  XXXX  |
      |  XXXX  |
      |        |   where "X" is the central 50% of the image.
       --------
    Args:
    image: 3-D array of shape [height, width, depth]
    central_fraction: float (0, 1], fraction of size to crop
    Raises:
    ValueError: if central_crop_fraction is not within (0, 1].
    Returns:
    3-D array
    """
    if central_fraction <= 0.0 or central_fraction > 1.0:
        raise ValueError('central_fraction must be within (0, 1]')
    if central_fraction == 1.0:
        return image

    img_shape = image.shape
    fraction_offset = int(1 / ((1 - central_fraction) / 2.0))
    bbox_w_start = np.divide(img_shape[1], fraction_offset)
    bbox_w_size = img_shape[1] - bbox_w_start * 2
    image = image[:,int(bbox_w_start):int(bbox_w_start+bbox_w_size)]
    return image


# In[ ]:


#preprocessing depth
def depth_preprocessing(mat, width=224, height=224):
    img = central_crop(mat, 0.875)
    img = resize(img,height,width)
    mat = milimeters(img)
    mat = escale_data(mat)
    img = viridis(mat)
    return img

