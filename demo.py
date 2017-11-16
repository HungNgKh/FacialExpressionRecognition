import cv2
import tensorflow as tf
import numpy as np
import os
import model_training.data_manager as dm

LABELS = np.array(['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])
CONFIDENT_LIMIT = 0.8
ICON_PATHS = [
    'saved/assets/icons/angry.jpeg',
    'saved/assets/icons/disgust.jpeg',
    'saved/assets/icons/fear.png',
    'saved/assets/icons/happy.jpeg',
    'saved/assets/icons/sad.jpeg',
    'saved/assets/icons/surprise.jpeg',
    'saved/assets/icons/neutral.png'
]

META_PATH = '/home/khanhhung/deeplearning/saved/FER/best_model/best-model.meta'
CHECKPOINT_PATH = '/home/khanhhung/deeplearning/saved/FER/best_model'
PREPROCESS_VALUE = np.load('model_training/pre.npy')
OUTPUT_GRAPH = '/home/khanhhung/deeplearning/saved/FER/best_model/best_model.pb'


def preprocess_image(img, pre):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = np.reshape(img, [48, 48, 1])
    img = img.astype(np.float64)
    img /= 255
    img -= pre[0]
    img /= pre[1]
    img = dm.CROP_CENTER(img)
    img = np.reshape(img, [42, 42, 1])
    return img



class ModelServer:

    def __init__(self, filename):
        self.__face_cascade = cv2.CascadeClassifier('/home/khanhhung/opencv-3.2.0/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
        self.graph = self.__load_graph(filename)
        self.sess = tf.Session(graph=self.graph)
        assert self.sess._closed == False

    def __load_graph(self, filename):

        if not tf.gfile.Exists(filename):
            raise AssertionError("Import directory isn't exist")

        graph_def = tf.GraphDef()
        with tf.gfile.GFile(filename, 'rb') as f:

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
            tf.import_graph_def(graph_def, name='prefix')

        return graph


    def load(self):
        self.__x_tensor = self.graph.get_tensor_by_name('prefix/Placeholder:0')
        # self.y_tensor = tf.get_default_graph().get_tensor_by_name('Placeholder_1:0')
        self.__phase_tensor = self.graph.get_tensor_by_name('prefix/Placeholder_2:0')
        self.__pred_tensor = self.graph.get_tensor_by_name('prefix/Softmax:0')
        # self.loss_tensor = tf.get_default_graph().get_tensor_by_name('Mean:0')

    def face_detect(self, img):
        # eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.__face_cascade.detectMultiScale(gray, 1.3, 5)
        print faces
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return [i for i in faces if i[2] >= 150 and i[3] >= 150]

    def predict(self, imgs):

        predicts = self.sess.run(self.__pred_tensor, feed_dict={self.__x_tensor : imgs, self.__phase_tensor: False})
        return predicts



if __name__ == "__main__":

    icons = []
    for path in ICON_PATHS:
        img = cv2.imread(path)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_NEAREST)
        icons.append(img)


    model = ModelServer(OUTPUT_GRAPH)
    model.load()

    # cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 400)
    #
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    # cv2.namedWindow('demo', cv2.WINDOW_NORMAL)

    # cv2.moveWindow('demo', 400, 100)

    # cv2.namedWindow('demo', cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty('demo', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cap.set(cv2.CAP_PROP_FPS, 60)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:

        ret, frame = cap.read()

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # print np.shape(gray)

        if ret == True:
            frame = np.array(frame[:,::-1,:])
            face_imgs = []
            faces = model.face_detect(frame)
            result = []

            for (x, y, w, h) in faces:
                face_imgs.append(preprocess_image(frame[y:y + h, x:x + w], PREPROCESS_VALUE))
            if np.size(faces, 0) > 0:
                result = model.predict(face_imgs)

            i = 0
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                index = np.argmax(result[i], 0)
                if result[i, index] > CONFIDENT_LIMIT:
                    frame[y:y+32, x:x+32] = icons[index]
                else:
                    frame[y:y + 32, x:x + 32] = icons[6]
                i += 1


            cv2.imshow('demo', frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    print("Estimated frames per second : {:.6f}".format(fps));
    cap.release()
    cv2.destroyAllWindows()