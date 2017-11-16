import numpy as np
import tensorflow as tf


META_PATH = '/home/khanhhung/deeplearning/saved/FER/best_model/best-model.meta'
CHECKPOINT_PATH = '/home/khanhhung/deeplearning/saved/FER/best_model'

OUTPUT_GRAPH = '/home/khanhhung/deeplearning/saved/FER/best_model/best_model.pb'


def freeze_model():
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(META_PATH, clear_devices=True)

        # We restore the weights
        saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_PATH))
        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            ['Softmax']  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(OUTPUT_GRAPH, "wb") as f:
            f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))


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

    for op in graph.get_operations():
        print op.name
    # freeze_model()