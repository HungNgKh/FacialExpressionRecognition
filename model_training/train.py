from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import data_manager as dmg
from os.path import exists

import model as md



"""
# Load MNIST data
from tensorflow.examples.tutorials.mnist import input_data

dataset = input_data.read_data_sets("MNIST_data", one_hot=True)
"""


KEEP_PROB = 0.80
TRAINING_DATA_PATH = '../../datas/emotion/training'
VALIDATION_DATA_PATH = '../../datas/emotion/validation'
TEST_DATA_PATH = '../../datas/emotion/public_test'
SAVE_PATH = '../saved/model/emotion-model'
META_PATH = '../saved/model/emotion-model.meta'
CHECKPOINT_PATH = '../saved/model'
RECORD_PATH = '../saved/recorded'
LOG_PATH = '../saved/Log'

DISPLAY_STEP = 10 # iters
SAVE_STEP = 10  # epochs

pending_loss_rc = []
pending_acc_rc = []


def restore_model(sess):
    if (exists(META_PATH)):
        new_saver = tf.train.new_saver = tf.train.import_meta_graph(META_PATH)
        new_saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_PATH))
    else:
        sess.run(tf.global_variables_initializer())



def record():
    global pending_loss_rc
    global pending_acc_rc
    pending_loss_rc = np.array(pending_loss_rc)
    pending_acc_rc = np.array(pending_acc_rc)
    # save accuracy
    if (exists(RECORD_PATH + '/accuracy.npy') == False):
        np.save(RECORD_PATH + '/accuracy.npy', pending_acc_rc)
    else:
        arr = np.load(RECORD_PATH + '/accuracy.npy')
        arr = np.append(arr, pending_acc_rc, 0)
        # arr = np.reshape(arr, [-1, 3])
        np.save(RECORD_PATH + '/accuracy.npy', arr)

    # save loss
    if (exists(RECORD_PATH + '/loss.npy') == False):
        np.save(RECORD_PATH + '/loss.npy', pending_loss_rc)
    else:
        arr = np.load(RECORD_PATH + '/loss.npy')
        arr = np.append(arr, pending_loss_rc, 0)
        # arr = np.reshape(arr, [-1, 2])
        np.save(RECORD_PATH + '/loss.npy', arr)





def train_completed():
    global pending_loss_rc
    last_val_acc = np.array(pending_loss_rc).T[1]
    if (exists(RECORD_PATH + '/loss.npy') == True):
        val_loss = np.load(RECORD_PATH + '/loss.npy').T[1]
        size = np.size(val_loss)
        if (size >= SAVE_STEP and np.size(last_val_acc) == SAVE_STEP):
            # prev_loss_bg = size - SAVE_STEP
            # last_lost_bg = prev_loss_bg + SAVE_STEP
            prev_avg_loss = val_loss[size - 1]
            last_avg_loss = last_val_acc[SAVE_STEP - 1]
            print("\nPrev loss : " + "{:.6f}".format(prev_avg_loss) + "\nLast loss : " + "{:.6f}".format(last_avg_loss))
            return last_avg_loss > prev_avg_loss

    return False

# Trainer object
class Trainer:

    n_epoch = None
    model = None
    training_data = None
    validation_data = None
    test_data = None
    __n_iter_per_epoch = None




    #Constructor with param is epoch number and model
    def __init__(self, n_epoch = 1):
        self.n_epoch = n_epoch

    # load data and model
    def load(self):
        # create model
        self.model = md.ConvolutionalNetwork([48, 48, 1], 7, 3e-4)
        self.model.assemble()

        self.__correct_pred = tf.equal(tf.argmax(self.model.pred, 1), tf.argmax(self.model.y, 1))
        self.__accuracy = tf.reduce_mean(tf.cast(self.__correct_pred, tf.float32))
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(LOG_PATH + '/train', tf.get_default_graph())
        self.test_writer = tf.summary.FileWriter(LOG_PATH + '/test', tf.get_default_graph())

        # load model_training data
        print ("Loading data...")
        self.training_data = dmg.TrainingSet(128)
        self.training_data.load(TRAINING_DATA_PATH)
        mean, stddev = self.training_data.preprocess()
        self.training_data.shuffle()

        # load validation data
        self.validation_data = dmg.TestSet(100)
        self.validation_data.load(VALIDATION_DATA_PATH)
        self.validation_data.preprocess(mean, stddev)

        # load test data
        self.test_data = dmg.TestSet(100)
        self.test_data.load(TEST_DATA_PATH)
        self.test_data.preprocess(mean, stddev)

        self.__n_iter_per_epoch = self.training_data.size // self.training_data.batch_size
        print("Loading completed")


    # validate in a data set require session
    def validate(self, sess , data = dmg.TestSet(1)):
        avg_loss = 0
        n_iter = 0
        n_correct = 0
        data.reset()
        while(data.test_done() == False):
            batch = data.nextbatch()
            loss, corr, = sess.run([self.model.cost, self.__correct_pred],
                                feed_dict={self.model.x: batch.x, self.model.y: batch.y, self.model.keep_prob: 1.0})

            avg_loss += loss
            n_iter += 1
            n_correct += np.sum(corr, 0)

        avg_loss = avg_loss / n_iter
        accuracy = n_correct / data.size

        return avg_loss, accuracy


    # test in data set require session to run
    def test(self, sess , data = dmg.TestSet(1)):
        n_correct = 0
        data.reset()
        while (data.test_done() == False):
            batch = data.nextbatch()
            corr = sess.run(self.__correct_pred,
                                  feed_dict={self.model.x: batch.x, self.model.y: batch.y, self.model.keep_prob: 1.0})

            n_correct += np.sum(corr, 0)

        accuracy = n_correct / data.size

        return accuracy


    def log(self, current_epoch):
        global pending_loss_rc
        global pending_acc_rc
        pending_loss_rc = np.array(pending_loss_rc)
        pending_acc_rc = np.array(pending_acc_rc)

        # Plot loss to event file
        begin_step = current_epoch - np.size(pending_loss_rc, axis=0) + 1
        for i in range(np.size(pending_loss_rc, axis=0)):

            summary = tf.Summary(value=[tf.Summary.Value(tag='Loss', simple_value=pending_loss_rc[i][0])])
            self.train_writer.add_summary(summary, begin_step + i)

            summary = tf.Summary(value=[tf.Summary.Value(tag='Loss', simple_value=pending_loss_rc[i][1])])
            self.test_writer.add_summary(summary, begin_step + i)


        # Plot accuracy to event file
        begin_step = current_epoch - np.size(pending_acc_rc, axis=0) + 1
        for i in range(np.size(pending_acc_rc, axis=0)):

            summary = tf.Summary(value=[tf.Summary.Value(tag='Accuracy', simple_value=pending_acc_rc[i][0])])
            self.train_writer.add_summary(summary, begin_step + i)

            summary = tf.Summary(value=[tf.Summary.Value(tag='Accuracy', simple_value=pending_acc_rc[i][1])])
            self.test_writer.add_summary(summary, begin_step + i)


    # run the Trainer
    def run(self):

        with tf.Session() as sess:
            stop_flag = False

            saver = tf.train.Saver()


            # if(exists(META_PATH)):
            #     new_saver = tf.train.new_saver = tf.train.import_meta_graph(META_PATH)
            #     new_saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_PATH))
            # else:
            #     sess.run(tf.global_variables_initializer())


            # Initial some value to caculate loss and accuracy
            restore_model(sess)
            iter = sess.run(self.model.global_step)
            current_epoch = iter // self.__n_iter_per_epoch + 1

            epoch_accuracy = 0
            epoch_loss = 0

            # Training
            while stop_flag == False:
                batch = self.training_data.nextbatch()
                batch.x = dmg.data_augment(batch.x)

                _, loss, acc, summary = sess.run([self.model.optimizer, self.model.cost, self.__accuracy, self.merged],
                                        feed_dict={self.model.x: batch.x, self.model.y: batch.y, self.model.keep_prob: KEEP_PROB})


                epoch_accuracy += acc
                epoch_loss += loss
                iter += 1


                # Display model_training progress each num ber of iters
                if(iter % DISPLAY_STEP == 0):
                    print("Iter " + str(iter) + ", Epoch " + str(current_epoch) +", Batch loss = " + "{:.6f}".format(
                    loss) + ", Accuracy = " + "{:.6f}".format(acc))

                # Caculate after each epoch
                if (iter % self.__n_iter_per_epoch == 0):
                    # log
                    self.train_writer.add_summary(summary, current_epoch)
                    # Caculate loss and accuracy
                    epoch_accuracy = epoch_accuracy / self.__n_iter_per_epoch
                    epoch_loss = epoch_loss / self.__n_iter_per_epoch

                    # print result
                    print("==============================================================================")
                    print("Epoch " + str(current_epoch) + " finished!" + " , Loss = " + "{:.6f}".format(epoch_loss) + ", Accuracy = " + "{:.6f}".format(epoch_accuracy))
                    print("------------------------------------------------------------------------------")
                    print("Validating...")
                    val_loss, val_acc = self.validate(sess, self.validation_data)
                    print("Validation loss : " + "{:.6f}".format(val_loss) + " - Accuracy: " + "{:.6f}".format(val_acc))
                    # print("Testing...")
                    # test_acc = self.test(sess, self.test_data)
                    # print("Testing accuracy: " + "{:.6f}".format(test_acc))
                    print("==============================================================================\n")

                    # Save accuracy and loss
                    global pending_loss_rc
                    global pending_acc_rc


                    pending_acc_rc.append([epoch_accuracy, val_acc])
                    pending_loss_rc.append([epoch_loss, val_loss])

                    # stop when reach max perfomance and save model
                    stop_flag = train_completed()
                    if(current_epoch % SAVE_STEP == 0 and stop_flag == False):
                        print("Saving model...")
                        saver.save(sess, SAVE_PATH)
                        record()
                        self.log(current_epoch)

                        # clear pending record infomation
                        pending_loss_rc = []
                        pending_acc_rc = []
                        print("Model saved\n")


                    # rewind values
                    epoch_accuracy = 0
                    epoch_loss = 0
                    current_epoch += 1



# Run program
tr = Trainer(200)
tr.load()
tr.run()
print("Training finished")

