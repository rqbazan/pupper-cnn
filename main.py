try:
    import sys
    import os
    import random
    from glob import glob
    import cv2                
    import numpy as np
    from sklearn.datasets import load_files
    from keras.utils import np_utils
    import matplotlib.pyplot as plt
    from keras.applications.resnet50 import ResNet50
    from keras.preprocessing import image              
    from keras.applications.resnet50 import preprocess_input, decode_predictions 
    from keras.preprocessing.image import ImageDataGenerator
    from PIL import ImageFile
except ImportError as e:
    print(e)
    sys.exit(1)

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT_PATH = r'C:/Users/sciadmin/Documents/PupperCNN/Data'

dog_test_data_directory = os.path.join(ROOT_PATH, "dogImages/test")
dog_train_data_directory = os.path.join(ROOT_PATH, "dogImages/train")
dog_valid_data_directory = os.path.join(ROOT_PATH, "dogImages/valid")

human_data_directory = os.path.join(ROOT_PATH, "humanImages")

def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

train_files, train_targets = load_dataset(dog_train_data_directory)
valid_files, valid_targets = load_dataset(dog_valid_data_directory)
test_files, test_targets = load_dataset(dog_test_data_directory)

#dog_names = [item[20:-1] for item in sorted(glob("{}/*/".format(dog_train_data_directory)))]

def extractDogName(filepath):
    dogname = os.path.basename(filepath)[4:]
    return " ".join(dogname.split("_")).title()

dog_names = [extractDogName(item) for item in sorted(glob("{}/*".format(dog_train_data_directory)))]

print('There are %d total dog categories.' % len(dog_names))
print('There are %d total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

human_files = np.array(glob("{}/*/*".format(human_data_directory)))
random.shuffle(human_files)

print('There are %d total human images.' % len(human_files))

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

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
    return ((prediction <= 268) & (prediction >= 151)) 

"""
    Construyendo la CNN desde 0
"""

# train_tensors = paths_to_tensor(train_files).astype('float32')/255
# valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
# test_tensors = paths_to_tensor(test_files).astype('float32')/255


# datagen = ImageDataGenerator(
#     width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
#     height_shift_range=0.1,  # randomly shift images vertically (10% of total height)
#     horizontal_flip=True) # randomly flip images horizontally

# datagen.fit(train_tensors)


# from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
# from keras.layers import Dropout, Flatten, Dense, Activation
# from keras.models import Sequential
# from keras.layers.normalization import BatchNormalization

# model = Sequential()
# model.add(BatchNormalization(input_shape=(224, 224, 3)))
# model.add(Conv2D(filters=16, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(BatchNormalization())

# model.add(Conv2D(filters=32, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(BatchNormalization())

# model.add(Conv2D(filters=64, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(BatchNormalization())

# model.add(Conv2D(filters=128, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(BatchNormalization())

# model.add(Conv2D(filters=256, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(BatchNormalization())

# model.add(GlobalAveragePooling2D())

# model.add(Dense(133, activation='softmax'))

# model.summary()

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# """
#     Guardando el modelo
# """

# model.save_weights("saved_models/weights.bestaugmented.from_scratch.hdf5")


# """
#     Entrenando el modelo
# """

# from keras.callbacks import ModelCheckpoint  

# epochs = 10
# batch_size = 20

# checkpointer = ModelCheckpoint(filepath='saved_models/weights.bestaugmented.from_scratch.hdf5', 
#                                verbose=1, save_best_only=True)

# ### Using Image Augmentation
# model.fit_generator(datagen.flow(train_tensors, train_targets, batch_size=batch_size),
#                     validation_data=(valid_tensors, valid_targets), 
#                     steps_per_epoch=train_tensors.shape[0] // batch_size,
#                     epochs=epochs, callbacks=[checkpointer], verbose=1)



# model.load_weights('saved_models/weights.bestaugmented.from_scratch.hdf5')

# dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# # report test accuracy
# test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
# print('Test accuracy: %.4f%%' % test_accuracy)

# """
#     Más entrenamiento
# """

# batch_size = 20
# epochs = 5

# model.fit_generator(datagen.flow(train_tensors, train_targets, batch_size=batch_size),
#                     validation_data=(valid_tensors, valid_targets), 
#                     steps_per_epoch=train_tensors.shape[0] // batch_size,
#                     epochs=epochs, callbacks=[checkpointer], verbose=1)


# # In[136]:


# batch_size = 20
# epochs = 5

# model.fit_generator(datagen.flow(train_tensors, train_targets, batch_size=batch_size),
#                     validation_data=(valid_tensors, valid_targets), 
#                     steps_per_epoch=train_tensors.shape[0] // batch_size,
#                     epochs=epochs, callbacks=[checkpointer], verbose=1)


# # In[139]:


# batch_size = 64
# epochs = 5

# model.fit_generator(datagen.flow(train_tensors, train_targets, batch_size=batch_size),
#                     validation_data=(valid_tensors, valid_targets), 
#                     steps_per_epoch=train_tensors.shape[0] // batch_size,
#                     epochs=epochs, callbacks=[checkpointer], verbose=1)


# # ## Test the Model Again
# # After we test the model again we see that we improved test accuracy up to 51%

# # In[140]:


# # get index of predicted dog breed for each image in test set
# dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# # report test accuracy
# test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
# print('Test accuracy: %.4f%%' % test_accuracy)

"""
    Creando la CNN usando Transfer Learning
"""

bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
train_ResNet50 = bottleneck_features['train']
valid_ResNet50 = bottleneck_features['valid']
test_ResNet50 = bottleneck_features['test']


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adamax

# ResNet_model = Sequential()
# ResNet_model.add(GlobalAveragePooling2D(input_shape=train_ResNet50.shape[1:]))
# ResNet_model.add(Dense(133, activation='softmax'))

# ResNet_model.summary()

# ResNet_model.compile(loss='categorical_crossentropy', optimizer=Adamax(lr=0.002), metrics=['accuracy'])

# """
#     Guardando el modelo
# """

# ResNet_model.save_weights("saved_models/weights.best_adamax.ResNet50.hdf5")


# """
#     Entrenando el modelo
# """

from keras.callbacks import ModelCheckpoint  

# checkpointer = ModelCheckpoint(filepath='saved_models/weights.best_adamax.ResNet50.hdf5', 
#                                verbose=1, save_best_only=True)

# epochs = 30
# batch_size = 64

# ResNet_model.fit(train_ResNet50, train_targets, 
#           validation_data=(valid_ResNet50, valid_targets),
#           epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)


# # In[232]:


# opt = Adamax(lr=0.0002)
# epochs = 5
# batch_size = 64

# ResNet_model.fit(train_ResNet50, train_targets, 
#           validation_data=(valid_ResNet50, valid_targets),
#           epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)


# # ### Load the Model with the Best Validation Loss

# # In[230]:


# ### Load the model weights with the best validation loss.
# ResNet_model.load_weights('saved_models/weights.best_adamax.ResNet50.hdf5')

"""
    Testeando el modelo
"""

# # get index of predicted dog breed for each image in test set
# ResNet50_predictions = [np.argmax(ResNet_model.predict(np.expand_dims(feature, axis=0))) for feature in test_ResNet50]

# # report test accuracy
# test_accuracy = 100*np.sum(np.array(ResNet50_predictions)==np.argmax(test_targets, axis=1))/len(ResNet50_predictions)
# print('Test accuracy: %.4f%%' % test_accuracy)

"""
    Validando el modelo
"""

ResNet_model = Sequential()
ResNet_model.add(GlobalAveragePooling2D(input_shape=train_ResNet50.shape[1:]))
ResNet_model.add(Dense(133, activation='softmax'))

# ResNet_model.summary()
ResNet_model.compile(loss='categorical_crossentropy', optimizer=Adamax(lr=0.002), metrics=['accuracy'])
ResNet_model.load_weights('saved_models/weights.best_adamax.ResNet50.hdf5')

from extract_bottleneck_features import *

def ResNet50_predict_breed(img_path):
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    predicted_vector = ResNet_model.predict(bottleneck_feature)
    breed = dog_names[np.argmax(predicted_vector)]
    img = cv2.imread(img_path)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgplot = plt.imshow(cv_rgb)
    if dog_detector(img_path) == True:
        return print("The breed of dog is a {}".format(breed))
    else:
        return print("If this person were a dog, the breed would be a {}".format(breed))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def predict_breed(img_path):
    isDog = dog_detector(img_path)
    isPerson = face_detector(img_path)
    if isDog:
        print("Detected a dog")
        breed = ResNet50_predict_breed(img_path)
        return breed
    if isPerson:
        print("Detected a human face")
        breed = ResNet50_predict_breed(img_path)
        return breed
    else:
        print("No human face or dog detected")
        img = cv2.imread(img_path)
        cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgplot = plt.imshow(cv_rgb)

def main():
    predict_breed('{}/002.Afghan_hound/Afghan_hound_00081.jpg'.format(dog_train_data_directory))

if __name__ == '__main__':
    main()
    