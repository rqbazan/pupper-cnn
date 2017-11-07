import os
import cv2                
from glob import glob
import numpy as np
from sklearn.datasets import load_files
from keras.utils import np_utils
import matplotlib.pyplot as plt

ROOT_PATH = "D:\PupperCNN\Data"

dog_test_data_directory = os.path.join(ROOT_PATH, "dogImages/test")
dog_train_data_directory = os.path.join(ROOT_PATH, "dogImages/train")
dog_valid_data_directory = os.path.join(ROOT_PATH, "dogImages/valid")

human_data_directory = os.path.join(ROOT_PATH, "humanImages")

def load_dataset(path):
    print("load_dataset", path)
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# train_files, train_targets = load_dataset(dog_train_data_directory)
# valid_files, valid_targets = load_dataset(dog_valid_data_directory)
# test_files, test_targets = load_dataset(dog_test_data_directory)

# dog_names = [item[20:-1] for item in sorted(glob("{}/*/".format(dog_train_data_directory)))]

# print('There are %d total dog categories.' % len(dog_names))
# print('There are %d total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
# print('There are %d training dog images.' % len(train_files))
# print('There are %d validation dog images.' % len(valid_files))
# print('There are %d test dog images.'% len(test_files))

import random
random.seed(8675309)

human_files = np.array(glob("{}/*/*".format(human_data_directory)))
random.shuffle(human_files)

print('There are %d total human images.' % len(human_files))

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

human_files_short = human_files[:100]
dog_files_short = train_files[:100]

def main():
    pass

if __name__ == '__main__':
    main()
    