import os
from glob import glob


ROOT_PATH = r'C:/Users/sciadmin/Documents/PupperCNN/Data'

dog_test_data_directory = os.path.join(ROOT_PATH, "dogImages/test")
dog_train_data_directory = os.path.join(ROOT_PATH, "dogImages/train")
dog_valid_data_directory = os.path.join(ROOT_PATH, "dogImages/valid")

def extractDogName(filepath):
    dogname = os.path.basename(filepath)[4:]
    return " ".join(dogname.split("_")).title()

dog_names = [extractDogName(item) for item in sorted(glob("{}/*".format(dog_train_data_directory)))]

print(len(dog_names))
print(dog_names[0])