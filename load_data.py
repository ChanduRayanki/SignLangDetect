import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.model_selection import train_test_split



data_dir = r"E:\CHANDU\Projects\ComputerVisionProjects\SignLangDetect\asl_dataset"
def load_data(data_dir):
    # load data
    print("Loading data....")
    images=[]
    labels=[]
    label_map= {idx : label for idx, label in enumerate(os.listdir(data_dir))}
    #print("label_map: ",label_map)
    for i,label in enumerate(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir,label)
        for img in os.listdir(label_dir):
            img_path = os.path.join(label_dir,img)
            image = load_img(img_path)
            image = img_to_array(image)
            images.append(image)
            labels.append(label_map[i])
    images = np.array(images)
    labels = np.array(labels)
    print(labels)
    print("Done...")
    return images, labels

load_data(data_dir)