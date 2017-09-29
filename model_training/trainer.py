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



# Data path
TRAINING_DATA_PATH = '../../datas/emotion/train'
VALIDATION_DATA_PATH = '../../datas/emotion/public_test'
TEST_DATA_PATH = '../../datas/emotion/private_test'


# Model path
SAVE_PATH = '../saved/model/my-model'
BEST_MODEL_PATH = '../saved/best_model/best-model'
META_PATH = '../saved/model/my-model.meta'
CHECKPOINT_PATH = '../saved/model'
RECORD_PATH = '../saved/recorded'
LOG_PATH = '../saved/Log'

DISPLAY_STEP = 10 # iters
SAVE_STEP = 10  # epochs

pending_loss_rc = []
pending_acc_rc = []
pending_histogram = []


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





def better_performance():
    global pending_acc_rc
    last_val_acc = np.array(pending_acc_rc).T[1]
    if (exists(RECORD_PATH + '/loss.npy') == True):
        prev_val_acc = np.load(RECORD_PATH + '/accuracy.npy').T[1]
        prev_avg_acc = np.max(prev_val_acc[SAVE_STEP-1::SAVE_STEP], 0)
        last_avg_acc = last_val_acc[np.size(last_val_acc) - 1]
        print("Best performance: " + "{:.6f}".format(prev_avg_acc) + "\nLast performance: " + "{:.6f}".format(last_avg_acc))
        return last_avg_acc > prev_avg_acc

    return True


def train_stop():
    global pending_loss_rc
    last_val_loss = np.array(pending_loss_rc).T[0]
    if (exists(RECORD_PATH + '/loss.npy') == True):
        prev_val_loss = np.load(RECORD_PATH + '/loss.npy').T[0]
        prev_avg_loss = prev_val_loss[np.size(prev_val_loss, 0) - 1]
        last_avg_loss = last_val_loss[np.size(last_val_loss) - 1]
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
    def __init__(self):
        pass

    # load data and model
    def load(self):
        # create model
        self.model = md.ConvolutionalNetwork([42, 42, 1], 7, 1e-3)
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
        self.training_data.shuffle()

        # load validation data
        self.validation_data = dmg.TestSet(128)
        self.validation_data.load(VALIDATION_DATA_PATH)

        # load test data
        self.test_data = dmg.TestSet(128)
        self.test_data.load(TEST_DATA_PATH)

        # preprocess data
        if(exists('../saved/values/preprocess.npy')):
            self.training_data.preprocess()
            pre = np.load('../saved/values/preprocess.npy')
            self.validation_data.preprocess(pre[0], pre[1])
            self.test_data.preprocess(pre[0], pre[1])
        else:
            mean, stddev = self.training_data.preprocess()
            np.save('../saved/values/preprocess.npy', np.array([mean, stddev]))
            self.validation_data.preprocess(mean, stddev)
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
            aug_data = dmg.data_augment(batch.x, [dmg.CROP_CENTER])
            loss, corr, = sess.run([self.model.cost, self.__correct_pred],
                                feed_dict={self.model.x: aug_data, self.model.y: batch.y, self.model.is_training: False})

            avg_loss += loss
            n_iter += 1
            n_correct += np.sum(corr, 0)

        summary = sess.run(self.merged,
                           feed_dict={self.model.x: dmg.data_augment(data.datas[:1], [dmg.CROP_CENTER]), self.model.y: data.labels[:1], self.model.is_training: False})

        avg_loss = avg_loss / n_iter
        accuracy = n_correct / data.size

        return avg_loss, accuracy, summary


    # test in data set require session to run
    def test(self, sess , data = dmg.TestSet(1)):
        n_correct = np.array([0, 0, 0, 0, 0, 0, 0])
        data.reset()
        while (data.test_done() == False):
            batch = data.nextbatch()
            aug_data = dmg.data_augment(batch.x, [dmg.CROP_CENTER])
            pred = sess.run(self.model.pred,
                                  feed_dict={self.model.x: aug_data, self.model.y: batch.y, self.model.is_training: False})
            one_hot_pred = np.zeros_like(pred)
            one_hot_pred[range(np.size(pred, 0)), np.argmax(pred, 1)] = 1
            n_correct = np.add(n_correct, np.sum(np.multiply(one_hot_pred, batch.y), axis= 0))

        return n_correct


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


        # Plot variable histogram to event file
        begin_step = current_epoch - np.size(pending_histogram, axis=0) + 1
        for i in range(np.size(pending_acc_rc, axis=0)):
            self.train_writer.add_summary(pending_histogram[i], begin_step + i)




    # run the Trainer
    def run(self):

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        with tf.Session(config=config) as sess:

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
            print(iter)
            current_epoch = iter // self.__n_iter_per_epoch + 1

            epoch_accuracy = 0
            epoch_loss = 0

            # Training
            while stop_flag == False:
                batch = self.training_data.nextbatch()
                aug_data = dmg.data_augment(batch.x, [dmg.RANDOM_CROP, dmg.ROTATE, dmg.FLIP_MIRROR])

                _, loss, acc = sess.run([self.model.optimizer, self.model.cost, self.__accuracy],
                                        feed_dict={self.model.x: aug_data, self.model.y: batch.y, self.model.is_training: True})


                epoch_accuracy += acc
                epoch_loss += loss
                iter += 1

                # Display model_training progress each num ber of iters
                if(iter % DISPLAY_STEP == 0):
                    print("Iter " + str(iter) + ", Epoch " + str(current_epoch) +", Batch loss = " + "{:.6f}".format(
loss) + ", Accuracy = " + "{:.6f}".format(acc))

                # Caculate after each epoch
                if (iter % self.__n_iter_per_epoch == 0):

                    # Caculate loss and accuracy
                    epoch_accuracy = epoch_accuracy / self.__n_iter_per_epoch
                    epoch_loss = epoch_loss / self.__n_iter_per_epoch

                    # print result
                    print("==============================================================================")
                    print("Epoch " + str(current_epoch) + " finished!" + " , Loss = " + "{:.6f}".format(epoch_loss) + ", Accuracy = " + "{:.6f}".format(epoch_accuracy))
                    print("------------------------------------------------------------------------------")
                    print("Validating...")
                    val_loss, val_acc, summary = self.validate(sess, self.validation_data)
                    print("Validation loss : " + "{:.6f}".format(val_loss) + " - Accuracy: " + "{:.6f}".format(val_acc))
                    # print("Testing...")
                    # test_acc = self.test(sess, self.test_data)
                    # print("Testing accuracy: " + "{:.6f}".format(test_acc))
                    print("==============================================================================\n")

                    # Save accuracy and loss
                    global pending_loss_rc
                    global pending_acc_rc
                    global pending_histogram

                    pending_acc_rc.append([epoch_accuracy, val_acc])
                    pending_loss_rc.append([epoch_loss, val_loss])
                    pending_histogram.append(summary)

                    # stop when reach max perfomance and save model
                    # stop_flag = better_performance()
                    if(current_epoch % SAVE_STEP == 0):
                        print("Saving model...")
                        saver.save(sess, SAVE_PATH)
                        self.log(current_epoch)

                        if (better_performance()):
                            print("Replacing best model...")
                            saver.save(sess, BEST_MODEL_PATH)
                            print("best model saved")

                        # stop_flag = train_stop()

                        record()
                        # clear pending record infomation
                        pending_loss_rc = []
                        pending_acc_rc = []
                        pending_histogram = []
                        print("Model saved\n")


                    # rewind values
                    epoch_accuracy = 0
                    epoch_loss = 0
                    current_epoch += 1




