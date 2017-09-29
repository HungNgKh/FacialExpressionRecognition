import tensorflow as tf
import tensorflow.contrib.layers as layer
import numpy as np
from os.path import exists

from tensorflow.contrib.keras.python.keras.initializers import one

SAVED_MODEL = '../saved/model/my-model.meta'

# SVM hinge loss with margin = 1
def hinge_loss(pred, labels):
    true_classes = tf.argmax(labels, 1)
    idx_flattened = tf.range(0, tf.shape(pred)[0]) * tf.shape(pred)[1] + tf.cast(true_classes, dtype=tf.int32)

    true_scores = tf.cast(tf.gather(tf.reshape(pred, [-1]),
                            idx_flattened), dtype=tf.float32)

    L = tf.nn.relu((1 + tf.transpose(tf.nn.bias_add(tf.transpose(pred), tf.negative(true_scores)))) * (1 - labels))

    final_loss = tf.reduce_mean(tf.reduce_sum(L,axis=1))
    return final_loss


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
        self.is_training = tf.placeholder(tf.bool)


        # layer parameters

    # with tf.device("/cpu:0"):
        # conv1-1 layer
        self.weights['WC1'] = tf.get_variable(name='conv1_1w', shape=[3, 3, 1, 32], initializer=layer.xavier_initializer())
        self.bias['BC1'] = tf.get_variable(name='conv1_1_b', shape=[32], initializer=tf.zeros_initializer)
        layer_summaries(self.weights['WC1'], self.bias['BC1'], 'conv1-1')

        # conv1-2 layer
        self.weights['WC2'] = tf.get_variable(name='conv1_2w', shape=[3, 3, 32, 32], initializer=layer.xavier_initializer())
        self.bias['BC2'] = tf.get_variable(name='conv1_2_b', shape=[32], initializer=tf.zeros_initializer)
        layer_summaries(self.weights['WC2'], self.bias['BC2'], 'conv1-2')

        # conv2-1 layer
        self.weights['WC3'] = tf.get_variable(name='conv2_1_w', shape=[3, 3, 32, 64], initializer=layer.xavier_initializer())
        self.bias['BC3'] = tf.get_variable(name='conv2_1_b', shape=[64], initializer=tf.zeros_initializer)
        layer_summaries(self.weights['WC3'], self.bias['BC3'], 'conv2_1')

        # conv2-2 layer
        self.weights['WC4'] = tf.get_variable(name='conv2_2_w', shape=[3, 3, 64, 64], initializer=layer.xavier_initializer())
        self.bias['BC4'] = tf.get_variable(name='conv2_2_b', shape=[64], initializer=tf.zeros_initializer)
        layer_summaries(self.weights['WC4'], self.bias['BC4'], 'conv2_2')



    # with tf.device("/cpu:0"):
        # conv3-1 layer
        self.weights['WC5'] = tf.get_variable(name='conv3_1_w', shape=[3, 3, 64, 128], initializer=layer.xavier_initializer())
        self.bias['BC5'] = tf.get_variable(name='conv3_1_b', shape=[128], initializer=tf.zeros_initializer)
        layer_summaries(self.weights['WC5'], self.bias['BC5'], 'conv3_1')

        # conv3-2 layer
        self.weights['WC6'] = tf.get_variable(name='conv3_2_w', shape=[3, 3, 128, 128], initializer=layer.xavier_initializer())
        self.bias['BC6'] = tf.get_variable(name='conv3_2_b', shape=[128], initializer=tf.zeros_initializer)
        layer_summaries(self.weights['WC6'], self.bias['BC6'], 'conv3_2')

        # conv3-3 layer
        self.weights['WC7'] = tf.get_variable(name='conv3_3_w', shape=[3, 3, 128, 128], initializer=layer.xavier_initializer())
        self.bias['BC7'] = tf.get_variable(name='conv3_3_b', shape=[128], initializer=tf.zeros_initializer)
        layer_summaries(self.weights['WC7'], self.bias['BC7'], 'conv3_3')




    # with tf.device("/cpu:0"):
        # conv4-1 layer
        self.weights['WC8'] = tf.get_variable(name='conv4_1_w', shape=[3, 3, 128, 256],initializer=layer.xavier_initializer())
        self.bias['BC8'] = tf.get_variable(name='conv4_1_b', shape=[256], initializer=tf.zeros_initializer)
        layer_summaries(self.weights['WC8'], self.bias['BC8'], 'conv4_1')

        # conv4-2 layer
        self.weights['WC9'] = tf.get_variable(name='conv4_2_w', shape=[3, 3, 256, 256],initializer=layer.xavier_initializer())
        self.bias['BC9'] = tf.get_variable(name='conv4_2_b', shape=[256], initializer=tf.zeros_initializer)
        layer_summaries(self.weights['WC9'], self.bias['BC9'], 'conv4_2')

        # conv4-3 layer
        self.weights['WC10'] = tf.get_variable(name='conv4_3_w', shape=[3, 3, 256, 256],initializer=layer.xavier_initializer())
        self.bias['BC10'] = tf.get_variable(name='conv4_3_b', shape=[256], initializer=tf.zeros_initializer)
        layer_summaries(self.weights['WC10'], self.bias['BC10'], 'conv4_3')


    # with tf.device("/cpu:0"):
        # fc1 layer
        self.weights['WC11'] = tf.get_variable(name='fc1_w', shape=[3 * 3 * 256, 2048], initializer=layer.xavier_initializer())
        self.bias['BC11'] = tf.get_variable(name='fc1_b', shape=[2048], initializer=tf.zeros_initializer)
        layer_summaries(self.weights['WC11'], self.bias['BC11'], 'fc1')

        # fc3 layer output
        self.weights['WC12'] = tf.get_variable(name='fc2_w', shape=[2048, self.__n_class], initializer=layer.xavier_initializer())
        self.bias['BC12'] = tf.get_variable(name='fc2_b', shape=[self.__n_class], initializer=tf.zeros_initializer)
        layer_summaries(self.weights['WC12'], self.bias['BC12'], 'fc2')



    # Convert to feature value
    def __conv2d(self, x, w, b, stride=1):
        x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding="SAME")
        x = tf.nn.bias_add(x, b)
        return x

    # batch normalize
    def _batch_norm(self, inputs, scope_bn):
        bn_train = layer.batch_norm(inputs, decay=0.999, center=True, scale=True,
                              is_training=True,
                              updates_collections= None,
                              reuse=None,  # is this right?
                              trainable=True,
                              scope=scope_bn)

        bn_inference = layer.batch_norm(inputs, decay=0.999, center=True, scale=True,
                                  is_training=False,
                                  updates_collections=None,
                                  reuse=True,  # is this right?
                                  trainable=True,
                                  scope=scope_bn)

        z = tf.cond(self.is_training, lambda: bn_train, lambda: bn_inference)
        return z

    #dropout
    def _dropout(self, inputs, keep_prob):
        return tf.cond(self.is_training, lambda: tf.nn.dropout(inputs,keep_prob), lambda: tf.nn.dropout(inputs,1.0))



    # Create model
    def assemble(self):
        # Reshape input picture (48 x 48)
        # reshaped_x = tf.reshape(self.x, shape=[-1, 48, 48, 1])
        # image_preprocess(reshaped_x)
        # batch_size = tf.shape(self.x)

        # Convolution layer
        # conv1_1
        conv1_r = self.__conv2d(self.x, self.weights['WC1'], self.bias['BC1'])
        conv1_r = self._batch_norm(conv1_r, 'conv1-1')
        conv1_r = tf.nn.relu(conv1_r)

        # conv1_2
        conv1 = self.__conv2d(conv1_r, self.weights['WC2'], self.bias['BC2'])
        conv1 = self._batch_norm(conv1, 'conv1-2')
        conv1 = tf.nn.relu(conv1)

        #addition
        conv1 = tf.add(conv1_r, conv1)

        # Max Pooling (down-sampling)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")



        # conv2_1
        conv2_r = self.__conv2d(conv1, self.weights['WC3'], self.bias['BC3'])
        conv2_r = self._batch_norm(conv2_r, 'conv2-1')
        conv2_r = tf.nn.relu(conv2_r)

        # conv2_2
        conv2 = self.__conv2d(conv2_r, self.weights['WC4'], self.bias['BC4'])
        conv2 = self._batch_norm(conv2, 'conv2-2')
        conv2 = tf.nn.relu(conv2)

        # addition
        conv2 = tf.add(conv2_r, conv2)

        # Max Pooling (down-sampling)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")



        # conv3_1
        conv3_r = self.__conv2d(conv2, self.weights['WC5'], self.bias['BC5'])
        conv3_r = self._batch_norm(conv3_r, 'conv3-1')
        conv3_r = tf.nn.relu(conv3_r)

        # conv3_2
        conv3 = self.__conv2d(conv3_r, self.weights['WC6'], self.bias['BC6'])
        conv3 = self._batch_norm(conv3, 'conv3-2')
        conv3 = tf.nn.relu(conv3)

        # conv3_3
        conv3 = self.__conv2d(conv3, self.weights['WC7'], self.bias['BC7'])
        conv3 = self._batch_norm(conv3, 'conv3-3')
        conv3 = tf.nn.relu(conv3)

        # addition
        conv3 = tf.add(conv3_r, conv3)

        # Max Pooling (down-sampling)
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")



        # conv3_1
        conv4_r = self.__conv2d(conv3, self.weights['WC8'], self.bias['BC8'])
        conv4_r = self._batch_norm(conv4_r, 'conv4-1')
        conv4_r = tf.nn.relu(conv4_r)

        # conv3_2
        conv4 = self.__conv2d(conv4_r, self.weights['WC9'], self.bias['BC9'])
        conv4 = self._batch_norm(conv4, 'conv4-2')
        conv4 = tf.nn.relu(conv4)

        # conv3_3
        conv4 = self.__conv2d(conv4, self.weights['WC10'], self.bias['BC10'])
        conv4 = self._batch_norm(conv4, 'conv4-3')
        conv4 = tf.nn.relu(conv4)

        # addition
        conv4 = tf.add(conv4_r, conv4)

        # Max Pooling (down-sampling)
        conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


        # Fully connected layer
        # Reshape conv3 output to fit fully connected layer input
        fc = tf.reshape(conv4, [-1, self.weights['WC11'].get_shape().as_list()[0]])

        # fc1
        fc1 = tf.add(tf.matmul(fc, self.weights['WC11']), self.bias['BC11'])
        fc1 = self._batch_norm(fc1, 'fc1')
        fc1 = tf.nn.relu(fc1)
        fc1 = self._dropout(fc1, 0.6)


        # fc2
        fc2 = tf.add(tf.matmul(fc1, self.weights['WC12']), self.bias['BC12'])



        # # model prediction
        self.pred = tf.nn.softmax(fc2)


        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=fc2),0)

        self.global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)
        momentum = tf.constant(0.9)

        self.optimizer = tf.train.MomentumOptimizer(self.__learning_rate, momentum=momentum,
                                                        use_nesterov=True).minimize(self.cost, self.global_step)


