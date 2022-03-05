import tensorflow as tf
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sn



if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num of GPUs Available:", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    data_path = r"dataset_npy"
    outpath = r"result"
    
    # load data and labels
    color_images = np.load(os.path.join(data_path, 'data_images.npy'))
    labels = np.load(os.path.join(data_path, 'data_labels.npy'))

    # split the data and its labels for training and testing purpose
    Xtrain, Xtest, ytrain, ytest = train_test_split(color_images, labels, test_size=0.3, shuffle=True, random_state=42)
    
    # Normalize: 0,255 -> 0,1
    Xtrain, Xtest = Xtrain / 255.0, Xtest / 255.0
    
    # instantiates the MobileNetV2 architecture without the last layer
    # 1) instantiates the MobileNetV2 architecture
    mobile = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(96, 96, 3), weights='imagenet')
    model = tf.keras.Model(inputs=mobile.input, outputs=mobile.layers[-2].output)
    #model.summary()
    
    # make loaded layers non-trainable
    for layer in model.layers:
        layer.trainable = False
    model.summary()
    
    # apply the model to training and test images to extract the features for classical ML methods
    features_extracted_train = model.predict(Xtrain)
    features_extracted_test = model.predict(Xtest)

    # flatten the features
    new_Xtrain = features_extracted_train.reshape(features_extracted_train.shape[0], -1)
    new_Xtest = features_extracted_test.reshape(features_extracted_test.shape[0], -1)
    print(new_Xtrain.shape)
    
    # load pre trained scikit models
    model_RC = pickle.load(open(os.path.join('saved_model', 'model_RC_norm.sav'), 'rb'))
    model_LR = pickle.load(open(os.path.join('saved_model', 'model_LR_norm.sav'), 'rb'))
    model_SVC = pickle.load(open(os.path.join('saved_model', 'model_SVC_norm.sav'), 'rb'))
    
    # create a list of models to be tested
    models = [('Ridge Classifier', model_RC),
          ('Logistic Regressor', model_LR),
          ('SVC', model_SVC)]
    
    labels = ['with_mask', 'without_mask']
    # evaluate each model in turn
    accuracy_scores = []
    precision_scores = []
    f1_scores = []
    recall_scores = []
    confusion_matrices = []
    names = []
    for name, model in models:
        model.fit(new_Xtrain, ytrain)
        ypred = model.predict(new_Xtest) 
        a_score = accuracy_score(ytest, ypred)
        accuracy_scores.append(a_score)
        p_score = precision_score(ytest, ypred)
        precision_scores.append(p_score)
        f_score = f1_score(ytest, ypred)
        f1_scores.append(f_score)
        r_score = recall_score(ytest, ypred)
        recall_scores.append(r_score)
        confusion_matrices.append(confusion_matrix(ytest, ypred))
        names.append(name)
        
        print('%s: %f %f %f %f' % (name, a_score, p_score, f_score, r_score))
    
    results = np.array([names, accuracy_scores, precision_scores, f1_scores, recall_scores])
    np.save('Results 2.2_without_cv.txt', results)
    
    # plot confusion matrix

    for cm, name, score in zip(confusion_matrices, names, accuracy_scores):
        #print(f'{str(name)}:\n {np.array(cm)}')
        plt.figure(figsize=(4, 4))
        sn.heatmap(cm, annot=True, fmt='.0f', cmap = plt.cm.Blues, cbar=False)
        plt.yticks([0.5, 1.5], ['with_mask', 'without_mask'], va='center')
        plt.xticks([0.5, 1.5], ['with_mask', 'without_mask'], va='center')
        plt.title(f'{str(name)} (Accuracy score = {score:.2f})')
        plt.ylabel('predicted labels')
        plt.xlabel('true labels')
        plt.tight_layout()
        plt.savefig(os.path.join('result', 'Fig 2.2 confusion_matrix_'+str(name)+'.png'), dpi=200)
        plt.show()
        

        

    
    



