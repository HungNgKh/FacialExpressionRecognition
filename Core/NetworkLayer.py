
import tensorflow as tf
from tensorflow.contrib import layers



# Neural Network Layer
class NetworkLayer:

    def __init__(self, name):
        self._name = name


    @property
    def name(self):
        return self._name


    # return output tensor
    def apply(self, input):
        raise NotImplementedError



# 2d convolutional layer
class Conv2dlayer(NetworkLayer):

    def __init(self, name, kernel_shape, bias_shape, padding, stride=1, activation_func = None):
        NetworkLayer.__init__(self, name)
        with tf.variable_scope(name):
            self.weights = tf.get_variable('w', kernel_shape, initializer= layers.xavier_initializer_conv2d(uniform=False))
            self.bias = tf.get_variable('b', bias_shape, initializer=tf.random_normal_initializer)
        assert self.weights.name == name + '/w:0' and self.bias.name == name + '/b:0'
        tf.add_to_collection(self.weights.name, self.weights)
        tf.add_to_collection(self.bias.name, self.bias)

        self.__activation_func = activation_func
        self.__padding = padding
        self.__stride = stride



    def apply(self, input):
        conv = tf.nn.conv2d(input=input, filter=self.weights, strides=[1,self.__stride,self.__stride,1], padding=self.__padding)
        conv = tf.nn.bias_add(conv, self.bias)
        if self.__activation_func is None:
            return input
        else:
            return self.__activation_func(conv)


# fully-connected layer
class FullyConnectedLayer(NetworkLayer):

    def __init(self, name, weight_shape, bias_shape, activation_func = None):
        NetworkLayer.__init__(self, name)
        with tf.variable_scope(name):
            self.weights = tf.get_variable('w', weight_shape, initializer= layers.xavier_initializer(uniform=False))
            self.bias = tf.get_variable('b', bias_shape, initializer=tf.random_normal_initializer)
        assert self.weights.name == name + '/w:0' and self.bias.name == name + '/b:0'
        tf.add_to_collection(self.weights.name, self.weights)
        tf.add_to_collection(self.bias.name, self.bias)

        self.__activation_func = activation_func


    def apply(self, input):
        fc = tf.add(tf.matmul(input, self.weights), self.bias)
        if self.__activation_func is None:
            return input
        else:
            return self.__activation_func(fc)

# maxpool2d layer
class Maxpool2dLayer(NetworkLayer):

    def __init(self, name, size, stride, padding):
        NetworkLayer.__init__(self, name)
        self.__size = size
        self.__stride = stride
        self.__padding = padding


    def apply(self, input):
        return tf.nn.max_pool(input, ksize=[1,self.__size,self.__size,1], strides=[1,self.__stride,self.__stride,1],padding=self.__padding)


# Average pool 2d layer
class Avgpool2dLayer(NetworkLayer):

    def __init(self, name, size, stride, padding):
        NetworkLayer.__init__(self, name)
        self.__size = size
        self.__stride = stride
        self.__padding = padding


    def apply(self, input):
        return tf.nn.avg_pool(input, ksize=[1,self.__size,self.__size,1], strides=[1,self.__stride,self.__stride,1], padding=self.__padding)