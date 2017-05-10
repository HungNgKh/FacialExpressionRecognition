import tensorflow as tf
import numpy as np

# SVM hinge loss with margin = 1 input pred is a vector contain values from 0 to 1

def hinge_loss(pred, labels):
    pass

def layer_summaries(w, b, name):

    with tf.name_scope(name):
        tf.summary.histogram('weights', w)
        tf.summary.histogram('bias', b)


# Convolutional Neural Network Structure
class ConvolutionalNetwork:
    # Network Parameter
    __learning_rate = None
    __n_input = None
    __n_class = None

    # Construct Layers weights and bias
    def __init__(self, n_input, n_class, learning_rate):
        self.__n_input = n_input
        self.__n_class = n_class
        self.__learning_rate = learning_rate
        self.weights = dict()
        self.bias = dict()

        # Input
        input_shape = np.append([None], self.__n_input)
        self.x = tf.placeholder(tf.float32, input_shape)
        self.y = tf.placeholder(tf.float32, [None, self.__n_class])
        self.keep_prob = tf.placeholder(tf.float32)

        # 1st layer
        self.weights['WC1'] = tf.Variable(tf.random_normal([3, 3, 1, 16], stddev=0.47))
        self.bias['BC1'] = tf.Variable(tf.zeros([16]))
        layer_summaries(self.weights['WC1'], self.bias['BC1'], 'conv_layer1')

        # 2nd layer
        self.weights['WC2'] = tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=0.083))
        self.bias['BC2'] = tf.Variable(tf.zeros([32]))
        layer_summaries(self.weights['WC2'], self.bias['BC2'], 'conv_layer2')

        # 3rd layer
        self.weights['WC3'] = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.068))
        self.bias['BC3'] = tf.Variable(tf.zeros([64]))
        layer_summaries(self.weights['WC3'], self.bias['BC3'], 'conv_layer3')

        # 4th layer
        self.weights['WC4'] = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.058))
        self.bias['BC4'] = tf.Variable(tf.zeros([128]))
        layer_summaries(self.weights['WC4'], self.bias['BC4'], 'conv_layer4')

        # 5th layer
        self.weights['WD1'] = tf.Variable(tf.random_normal([3 * 3 * 128, 3072], stddev=0.042))
        self.bias['BD1'] = tf.Variable(tf.zeros([3072]))
        layer_summaries(self.weights['WD1'], self.bias['BD1'], 'fully_layer1')

        # 6th layer
        self.weights['OUT'] = tf.Variable(tf.random_normal([3072, self.__n_class], stddev=0.026))
        self.bias['OUT'] = tf.Variable(tf.zeros([self.__n_class]))
        layer_summaries(self.weights['OUT'], self.bias['OUT'], 'last_layer')


    # batch normalize
    def __batch_normalize(self, input):
        batch_mean, batch_var = tf.nn.moments(input, [0])
        scale1 = tf.Variable(tf.ones(input.get_shape().as_list()[-1]))
        beta1 = tf.Variable(tf.zeros(input.get_shape().as_list()[-1]))
        output = tf.nn.batch_normalization(input, batch_mean, batch_var, beta1, scale1, 1e-3)
        return output


    # Convert to feature value
    def __conv2d(self, x, w, b, stride=1):
        x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding="SAME")
        x = tf.nn.bias_add(x, b)
        return x



    # Create model
    def assemble(self):
        # Reshape input picture (48 x 48)
        # reshaped_x = tf.reshape(self.x, shape=[-1, 48, 48, 1])
        # image_preprocess(reshaped_x)

        # Convolution Layer
        conv1 = self.__conv2d(self.x, self.weights['WC1'], self.bias['BC1'])
        conv1 = self.__batch_normalize(conv1)
        conv1 = tf.nn.relu(conv1)

        # Max Pooling (down-sampling)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # Convolution Layer
        conv2 = self.__conv2d(conv1, self.weights['WC2'], self.bias['BC2'])
        conv2 = self.__batch_normalize(conv2)
        conv2 = tf.nn.relu(conv2)
        # Max Pooling (down-sampling)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # Convolution Layer
        conv3 = self.__conv2d(conv2, self.weights['WC3'], self.bias['BC3'])
        conv3 = self.__batch_normalize(conv3)
        conv3 = tf.nn.relu(conv3)
        # Max Pooling (down-sampling)
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # Convolution Layer
        conv4 = self.__conv2d(conv3, self.weights['WC4'], self.bias['BC4'])
        conv4 = self.__batch_normalize(conv4)
        conv4 = tf.nn.relu(conv4)
        # Max Pooling (down-sampling)
        conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # Fully connected layer
        # Reshape conv3 output to fit fully connected layer input
        fc1 = tf.reshape(conv4, [-1, self.weights['WD1'].get_shape().as_list()[0]])
        # 1st layer
        fc1 = tf.add(tf.matmul(fc1, self.weights['WD1']), self.bias['BD1'])
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, self.keep_prob)

        # Output, class prediction
        self.pred = tf.add(tf.matmul(fc1, self.weights['OUT']), self.bias['OUT'])

        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))

        self.global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)
        self.optimizer = tf.train.AdamOptimizer(self.__learning_rate).minimize(self.cost, self.global_step)
