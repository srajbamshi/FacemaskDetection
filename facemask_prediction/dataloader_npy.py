import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

orig_dataset_dir = r"Materials-20211215/facemask_dataset/dataset"
categories = ['with_mask', 'without_mask']

data = []
labels = []
image_size = 96

for category in categories:
    path = os.path.join(orig_dataset_dir, category)
    for img in os.listdir(path):
        if img != ".DS_Store":
            imgPath = os.path.join(path, img)
            image = cv2.imread(imgPath)[..., ::-1]  # read and convert BGR to RGB
            image = cv2.resize(image, (image_size, image_size))
            data.append(image)
            labels.append(category)

# convert categorical labels to numerical values
enc = LabelEncoder()
labels_encoded = enc.fit_transform(labels)
labels_encoded = np.array(labels_encoded)

# convert datatype of data to float32
data = np.array(data, dtype="float32")

# save data and labels to numpy file
np.save(os.path.join("dataset_npy", 'data_images.npy'), data)
np.save(os.path.join("dataset_npy", 'data_labels.npy'), labels_encoded)

# plot
im = data[0]
plt.imshow(im / 255)
plt.show()

