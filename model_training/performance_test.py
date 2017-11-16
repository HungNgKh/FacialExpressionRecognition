from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import data_manager as dmg
import model as md
from os.path import exists
from trainer import Trainer

labels = np.array(['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])

OUTPUT_GRAPH = '/home/khanhhung/deeplearning/saved/FER/best_model/best_model.pb'

TEST_SET = '/home/khanhhung/deeplearning/data/emotion/private_test'


def load_graph(file_name):

    if not tf.gfile.Exists(file_name):
        raise AssertionError("Import directory isn't exist")

    graph_def = tf.GraphDef()
    with tf.gfile.GFile(file_name, 'rb') as f:

        graph_def.ParseFromString(f.read())

    for node in graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name = 'prefix')

    return graph


if __name__ == "__main__":

    graph = load_graph(OUTPUT_GRAPH)

    x = graph.get_tensor_by_name('prefix/Placeholder:0')
    phase = graph.get_tensor_by_name('prefix/Placeholder_2:0')
    predict = graph.get_tensor_by_name('prefix/Softmax:0')
    test_data = dmg.TestSet(128)
    test_data.load(TEST_SET)
    pre = np.load('pre.npy')
    test_data.preprocess(pre[0], pre[1])

    with tf.Session(graph = graph) as sess:

        n_correct = np.array([0, 0, 0, 0, 0, 0, 0])
        test_data.reset()

        while (test_data.test_done() == False):
            batch = test_data.nextbatch()
            aug_data = dmg.data_augment(batch.x, [dmg.CROP_CENTER])
            pred = sess.run(predict,
                            feed_dict={x: aug_data, phase: False})
            one_hot_pred = np.zeros_like(pred)
            one_hot_pred[range(np.size(pred, 0)), np.argmax(pred, 1)] = 1
            n_correct = np.add(n_correct, np.sum(np.multiply(one_hot_pred, batch.y), axis=0))

        labs = np.sum(test_data.labels, 0)
        acc = np.sum(n_correct, 0) / np.size(test_data.datas, 0)
        for i in range(0, np.size(labels)):
            lab_acc = n_correct[i] / labs[i]
            print("Accuracy on " + labels[i] + " label:" + "{:.6f}".format(lab_acc))

        print("Accuracy : " + "{:.6f}".format(acc))

