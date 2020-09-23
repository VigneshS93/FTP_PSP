import cv2
import os
import glob
from matplotlib.pyplot import imread
data_dir = "/home/atipa/Project/motionArtifact/motionArtRed/Motion_Artifact/data/gt"
data_path = os.path.join(data_dir,'*g')
files = glob.glob(data_path)
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
images = load_images_from_folder(data_dir)
#Load data set
def read_dataset(path):
    images_path = f"{data_dir}/images"
    labels_path = f"{data_dir}/labels"

    images = np.zeros((320, 180, 180))
    labels = np.zeros((320, 180, 180))
images_train = np.zeros((320, 180, 180))
labels_train = np.zeros((320, 180, 180))
for i in range(320):
        img_file_path = f"sample_data/train/noisy/{i+1}.png"
        lbl_file_path = f"sample_data/train/groundtruth/{i+1}.png"
        
        images_train[i] = imread(img_file_path)
        labels_train[i] = imread(lbl_file_path)
images_test = np.zeros((80, 180, 180))
labels_test = np.zeros((80, 180, 180))
for i in range(321,400):
        img_file_path = f"sample_data/test/noisy/{i}.png"
        lbl_file_path = f"sample_data/test/groundtruth/{i}.png"
        
        images_test[i] = imread(img_file_path)
        labels_test[i] = imread(lbl_file_path)
train_groundtruth_path = 'sample_data/train/groundtruth'
train_noisy_path = 'sample_data/train/noisy'

test_groundtruth_path = 'sample_data/test/groundtruth'
test_noisy_path = 'sample_data/test/noisy'

train_label = os.listdir(train_groundtruth_path)
train_input = os.listdir(train_noisy_path)

test_label = os.listdir(test_groundtruth_path)
test_input = os.listdir(test_noisy_path)
