import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import cv2


# laod data and labels
color_images = np.load(r'data.npy')
labels = np.load(r'labels.npy')
gray_images = []
for color_image in color_images:
    gray_image = (cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY))
    gray_image= gray_image.flatten()
    gray_images.append(gray_image)

gray_images = np.array(gray_images)


# split the data and its labels for training and testing purpose
Xtrain, Xtest, ytrain, ytest = train_test_split(gray_images, labels, test_size=0.3, shuffle=True, random_state=42)


# create a list of models to be tested
model = RidgeClassifier()

kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
cv_results = cross_val_score(model, Xtrain, ytrain, cv=kfold, scoring='accuracy')
plt.boxplot(cv_results)