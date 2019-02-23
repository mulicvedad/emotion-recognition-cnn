import cv2
import os
import random
import numpy as np
from keras.preprocessing.image import img_to_array

# Emotions and labels
emotions = [('anger', 0), ('happy', 1), ('sadness', 2), ('surprise', 3), ('neutral', 4)]
# Cascade classifier for face recognition
classifier_path = 'cascade.xml'


class ImageDataManipulation:
    def __init__(self, src_folder, dest_folder, cascade_classifier_path=classifier_path,
                 subfolders=emotions):
        self.img_src_folder = src_folder
        self.training_data_folder = dest_folder
        self.emotions_folders = subfolders
        self.face_recognizer = self.load_face_recognizer(cascade_classifier_path)

    # Find and crop faces, transform to grayscale and save to 'dest_folder'
    def transform_images(self, resize_width, resize_height):
        for e_f in self.emotions_folders:
            sub_f = e_f[0]  # because e_f is tuple ('emotion', label), e.g. ('anger', 0)
            img_files = [f for f in os.listdir(os.path.join(self.img_src_folder, sub_f))
                         if os.path.isfile(os.path.join(self.img_src_folder, sub_f, f))]
            for img in img_files:
                src_img = cv2.imread(os.path.join(self.img_src_folder, sub_f, img))
                dest_folder = os.path.join(self.training_data_folder, sub_f)
                if not os.path.isdir(dest_folder):
                    os.mkdir(dest_folder)
                self.crop_face(image=src_img,
                               face_recognizer=self.face_recognizer,
                               resize_shape=(resize_width, resize_height),
                               save_to_file=os.path.join(dest_folder, img))

    # Return array of image file names and array of corresponding labels
    def load_training_data(self, subfolders=None,
                           num_img_files=1500, root_img_folder=None):
        if not root_img_folder:
            root_img_folder = self.training_data_folder
        if not subfolders:
            subfolders = self.emotions_folders
        num_imgs = num_img_files
        train_files = np.empty((num_imgs, 2), dtype=object)
        i = 0
        for x in subfolders:
            img_files = [f for f in os.listdir(os.path.join(root_img_folder, x[0])) if
                         os.path.isfile(os.path.join(os.path.join(root_img_folder, x[0], f)))]
            for img in img_files:
                train_files[i][0] = os.path.join(root_img_folder, x[0], img)  # relative file path
                train_files[i][1] = x[1]  # emotion label
                i += 1
        random.shuffle(train_files)  # shuffle the data
        return train_files[:, 0], train_files[:, 1]

    @staticmethod
    # Prepare training data. Load images from filesystem into array 'dataset'
    def preprocess_data(img_files, width=64, height=64, num_channels=1):
        dataset = np.ndarray(shape=(len(img_files), width, height, num_channels), dtype=np.float32)
        i = 0
        for file in img_files:
            img = cv2.imread(file)
            if img.shape[2] > 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (width, height))
            x = img_to_array(img)
            dataset[i] = x
            i += 1
        dataset = dataset / 255.
        return dataset

    # Prepare input image before prediction (crop, resize, grayscale)
    @staticmethod
    def prepare_img(img, width, height, grayscale=True):
        img = ImageDataManipulation.crop_face(img)
        if img is None:
            print('0 faces found in this picture')
            return None
        img = cv2.resize(img, (width, height))
        # To grayscale
        if len(img.shape) > 2 and img.shape[2] > 1 and grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img_to_array(img)
        dataset = np.empty(shape=[1, 64, 64, 1])
        dataset[0] = img
        dataset = dataset.astype('float32') / 255.
        return dataset

    @staticmethod
    def crop_face(image, face_recognizer=None, resize_shape=None, save_to_file=None):
        # To grayscale
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if not face_recognizer:
            face_recognizer = ImageDataManipulation.load_face_recognizer()
        faces = face_recognizer.detectMultiScale(
            img_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            return None
        crop_img = None
        for (x, y, w, h) in faces:
            crop_img = img_gray[y:y + h, x:x + w]
            if resize_shape:
                crop_img = cv2.resize(crop_img, resize_shape)
            if len(faces) > 1 and save_to_file:
                (folder, file) = os.path.split(save_to_file)
                file = str(random.randint(0, 100)) + file  # change the file name to avoid overwriting on filesystem
                save_to_file = os.path.join(folder, file)
            if save_to_file:
                cv2.imwrite(save_to_file, crop_img)
        return crop_img

    @staticmethod
    def load_face_recognizer(path=classifier_path):
        return cv2.CascadeClassifier(path)
