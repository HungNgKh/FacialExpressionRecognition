from __future__ import division
from PIL.Image import Image
from os.path import exists
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.io import loadmat
import numpy as np
import sklearn
import matplotlib.image as mpimg
from PIL import Image
import  cv2

data = loadmat('../datas/trainval/trainval/2008_000003.mat')

# print data.keys()


data = data['LabelMap']

img = Image.fromarray(data, 'L')
img.show()
print np.shape(data)
