
import tensorflow as tf
from  tensorflow.contrib import layers

SAVE_DIR = '../saved/model/'
MODEL_DIR = SAVE_DIR + 'my-model.ckpt'
GRAPH_DIR = SAVE_DIR + 'my-model.ckpt.meta'


class Model:

    __var_list = []
    # Create new model
    def __init__(self, name):
        self.__name = name

    @classmethod
    def load(cls, model_dir):
        pass


    @property
    def name(self):
        return self.__name

    # construct model variables and hyperparameter
    def construct(self, save_dir = None):
        raise NotImplementedError

    # construct op tensors and return output tensor
    def predict(self):
        return tf.get_collection('OUTPUT')

    # Save model
    def save(self, sess):
        saver = tf.train.Saver()
        saver.save(sess, SAVE_DIR + self.name + '.ckpt')













