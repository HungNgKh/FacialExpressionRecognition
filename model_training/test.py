from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import model as md
from os.path import exists
from trainer import Trainer

labels = np.array(['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])

SAVE_PATH = '../saved/model/my-model'
BEST_MODEL_PATH = '../saved/best_model/best-model'
META_PATH = '../saved/best_model/best-model.meta'
CHECKPOINT_PATH = '../saved/best_model'

def restore_model(sess):
    if (exists(META_PATH)):
        new_saver = tf.train.new_saver = tf.train.import_meta_graph(META_PATH)
        new_saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_PATH))
    else:
        sess.run(tf.global_variables_initializer())


tr = Trainer()
tr.load()
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.8
with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    restore_model(sess)
    corr = tr.test(sess, tr.test_data)
    labs = np.sum(tr.test_data.labels, 0)
    acc = np.sum(corr, 0) / np.size(tr.test_data.datas, 0)
    print("Accuracy on test set: " + "{:.6f}".format(acc))
    for i in range(0, np.size(labels)):
        lab_acc = corr[i] / labs[i]
        print("Accuracy on " + labels[i] + " label:" + "{:.6f}".format(lab_acc))



