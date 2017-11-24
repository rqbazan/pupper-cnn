try:
    import sys
    import os
    import logging as log
    import cv2
    import numpy as np
    from glob import glob
    from sklearn.datasets import load_files
    from keras.utils import np_utils    
    from keras.applications.resnet50 import ResNet50
    from keras.models import Sequential
    from keras.layers import GlobalAveragePooling2D, Dense
    from keras.optimizers import Adamax
    from keras.preprocessing import image    
    from keras.applications.resnet50 import preprocess_input
    from extract_bottleneck_features import *    
    from response import Response
except ImportError as e:
    log.error(e)
    sys.exit(1)

with open("dog_names") as f:
    dog_names = f.readlines()
dog_names = [name.strip() for name in dog_names]

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

"""
    Iniciando el modelo de la ResNet50 alimentado por la base de conocimiento en image-net
"""

ResNet50_model = ResNet50(weights='imagenet')


"""
    Preparando las imágenes para procesarlas en TensorFlow
"""

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]
    return np.vstack(list_of_tensors)

"""
    Realizando la predicción con ResNet50
"""

def ResNet50_predict_labels(img_path):
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return bool((prediction <= 268) and (prediction >= 151)) 

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
train_ResNet50 = bottleneck_features['train']
valid_ResNet50 = bottleneck_features['valid']
test_ResNet50 = bottleneck_features['test']

ResNet_model = Sequential()
ResNet_model.add(GlobalAveragePooling2D(input_shape=train_ResNet50.shape[1:]))
ResNet_model.add(Dense(133, activation='softmax'))

ResNet_model.compile(loss='categorical_crossentropy', optimizer=Adamax(lr=0.002), metrics=['accuracy'])
ResNet_model.load_weights('saved_models/weights.best_adamax.ResNet50.hdf5')

def ResNet50_predict_breed(img_path):
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    predicted_vector = ResNet_model.predict(bottleneck_feature)
    breed = dog_names[np.argmax(predicted_vector)]
    if dog_detector(img_path) == True:
        log.info("The breed of dog is a {}".format(breed))
    else:
        log.info("If this person were a dog, the breed would be a {}".format(breed))
    return breed

def predict_breed(img_path):
    response = Response()
    response.isDog = dog_detector(img_path)
    response.isHuman = face_detector(img_path)
    if response.isDog:
        log.info("Detected a dog")
        response.breed = ResNet50_predict_breed(img_path)
    elif response.isHuman:
        log.info("Detected a human face")
        response.breed = ResNet50_predict_breed(img_path)
    return response