import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import cv2


# laod data and labels
color_images = np.load(r'data.npy')
labels = np.load(r'labels.npy')

# convert RGB images to gray images
gray_images = []
for color_image in color_images:
    gray_image = (cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY))
    gray_images.append(gray_image)

gray_images = np.array(gray_images)

# flatten the images
flat_gray_images = gray_images.reshape(*gray_images.shape[:-2], -1)


# split the data and its labels for training and testing purpose
Xtrain, Xtest, ytrain, ytest = train_test_split(flat_gray_images, labels, test_size=0.3, shuffle=True, random_state=42)


# create a list of models to be tested
models = [('Ridge Classifier', RidgeClassifier()),
          ('Logistic Regressor', LogisticRegression(solver='liblinear')),
          ('SVC', SVC(gamma=0.01))]

# evaluate each model in turn
accuracy_scores = []
names = []
for name, model in models:
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    score = accuracy_score(ytest, ypred)
    accuracy_scores.append(score)
    names.append(name)
    print('%s: %f' % (name, score))

results = np.array([names, accuracy_scores])
np.save('results_without_cv.npy', results)



