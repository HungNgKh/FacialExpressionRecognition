from __future__ import division
from PIL.Image import Image
from os.path import exists
import tensorflow as tf
import numpy as np
import sklearn
import matplotlib.image as mpimg
from PIL import Image
from trainer import  Trainer
import data_manager as dm
import cv2


def face_detect(img_dir):
    face_cascade = cv2.CascadeClassifier('/home/khanhhung/opencv-3.2.0/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
    # eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    img = cv2.imread(img_dir)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return (img, faces)

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    pre = np.load('pre.npy')
    img = np.reshape(img, [48,48,1])
    img = img.astype(np.float64)
    img /= 255
    img -= pre[0]
    img /= pre[1]
    img = dm.CROP_CENTER(img)
    img = np.reshape(img, [42,42,1])
    return img

# plt.imshow(img, cmap = 'gray')
# plt.show()

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


# tr = Trainer()
# tr.load()
with tf.Session() as sess:
    restore_model(sess)

    x_tensor = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
    y_tensor = tf.get_default_graph().get_tensor_by_name('Placeholder_1:0')
    phase_tensor = tf.get_default_graph().get_tensor_by_name('Placeholder_2:0')
    pred_tensor = tf.get_default_graph().get_tensor_by_name('Softmax:0')
    loss_tensor = tf.get_default_graph().get_tensor_by_name('Mean:0')

    img, faces = face_detect('images/2.jpg')
    face_imgs = []
    for (x, y, w, h) in faces:
        face_imgs.append(preprocess_image(img[y:y+h,x:x+w]))

    size = np.size(face_imgs, 0)
    print size

    if(size > 0):
        pred = sess.run(pred_tensor,
                    feed_dict={x_tensor: face_imgs, y_tensor: np.zeros([size,7]), phase_tensor: False})


        print("Predict result : ")

        for i in range(size):
            out = np.argmax(pred[i], 0)
            print out
            print(str(labels[out]))


    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




