from __future__ import division
import numpy as np
from PIL import Image

# flip left to right an image (2D or 3D) randomly
def __random_fliplr(image):
    rd = np.random.randint(2)
    if(rd == 0):
        image = np.fliplr(image)
    return image


# return a rotated an image(2D or 3D) by max angle given
def __rotate(image, angle_begin, angle_end):
    shape = np.shape(image)
    img = np.reshape(image, [shape[0],shape[1]])
    img = Image.fromarray(img)
    rd_angle = np.random.randint(angle_begin, angle_end + 1)
    rotated = img.rotate(rd_angle)
    rotated = np.array(rotated)
    rotated = np.reshape(rotated, shape)
    return rotated


# return preprocessed image, input is a 3-D array
def image_preprocess(image):
    shape = np.shape(image)
    img = np.reshape(image, shape[0]*shape[1])
    img -= np.mean(img, axis=0)
    img /= np.std(img, axis=0)
    img = np.reshape(img, shape)
    return img


def preprocess(datas):
    datas -= np.mean(datas, axis=0)
    datas /= np.std(datas, axis=0)
    return datas

# return augmented images, input is a 4-D array
def data_augment(images):

    shape = np.shape(images)
    augmented_data = np.array([])

    for img in images:
        aug_img = __random_fliplr(img)
        aug_img = __rotate(aug_img, -45, 45)
        augmented_data = np.append(augmented_data, aug_img)

    return np.reshape(augmented_data, shape)


class Batch:

    size = None
    x = None
    y = None

    def __init__(self, x , y):
        self.x = x
        self.y = y
        self.size = np.size(x, 0)


# Data set structure
class DataSet:
    size = None
    batch_size = None

    datas = None
    labels = None
    _current_batch = None


    def __init__(self, batch_size):
        self.batch_size = batch_size



    # load data contain images and labels
    def load(self, folder_name):
        self.datas = np.load(folder_name + '/images.npy')
        self.datas = np.reshape(self.datas, [-1, 48, 48, 1])

        self.labels = np.load(folder_name + '/labels.npy')

        # self._datas = np.array(zip(images, labels))
        self.size = np.size(self.datas, 0)
        self._current_batch = 0


    def shuffle(self):
        datasset = zip(self.datas, self.labels)
        np.random.shuffle(datasset)
        self.datas, self.labels = zip(*datasset)

    def reset(self):
        self._current_batch = 0



# data set for model_training
class TrainingSet(DataSet):

    def preprocess(self):
        mean = np.mean(self.datas, axis=0)
        stddev = np.std(self.datas, axis=0)
        self.datas -= mean
        self.datas /= stddev
        return (mean, stddev)

    # return the next minibatch
    def nextbatch(self):
        if((self._current_batch + 1) * self.batch_size > self.size):
            self.shuffle()
            self._current_batch = 0

        index_from = self._current_batch * self.batch_size
        index_to = index_from + self.batch_size


        batch = Batch(self.datas[index_from:index_to], self.labels[index_from:index_to])
        self._current_batch += 1
        # batch.x = image_preprocess(batch.x)

        return batch


class TestSet(DataSet):

    def preprocess(self, mean, stddev):
        self.datas -= mean
        self.datas /= stddev

    # return the next minibatch
    def nextbatch(self):
        if ((self._current_batch + 1) * self.batch_size > self.size):
            index_from = self._current_batch * self.batch_size
            index_to = self.size

        else:
            index_from = self._current_batch * self.batch_size
            index_to = index_from + self.batch_size

        batch = Batch(self.datas[index_from:index_to], self.labels[index_from:index_to])
        self._current_batch += 1

        return batch

    #check when test is done
    def test_done(self):
        return self._current_batch * self.batch_size > self.size









