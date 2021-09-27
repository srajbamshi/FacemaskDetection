import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

datasetDir = r"dataset"
#print(os.listdir(datasetDir))

categories = ['without_mask', 'with_mask']
#print(categories)
#print(os.path.join(datasetDir, categories[0]))

data = []
labels = []
image_size = 96

for category in categories:
    path = os.path.join(datasetDir, category)
    for img in os.listdir(path):
        if (img != ".DS_Store"):
            #print("img:", img)
            imgPath = os.path.join(path, img)
            #print("imgpath:", imgPath)
            image = cv2.imread(imgPath)
            image = cv2.resize(image, (image_size, image_size))
            
            data.append(image)
            labels.append(category)    
            
# convert categorical labels to numerical values
enc = LabelEncoder()
labels_encoded = enc.fit_transform(labels)

# convert dtype of data to float32
data = np.array(data, dtype = "float32")
labels_encoded = np.array(labels_encoded)

# save data and labels to numpy file
np.save('data.npy', data) 
np.save('labels.npy', labels_encoded)