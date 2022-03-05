import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
import seaborn as sn
import cv2
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate, RandomizedSearchCV

import tensorflow as tf
import warnings
import pickle

warnings.filterwarnings("ignore")


def calculate_acc_prec_rec_f1(model, xtest, ytest):
    score = [accuracy_score(model.predict(xtest), ytest),
             precision_score(model.predict(xtest), ytest),
             f1_score(model.predict(xtest), ytest),
             recall_score(model.predict(xtest), ytest)]
    return score


def classify_RC(Xtrain, ytrain, Xtest, ytest, normalized=False):
    if normalized:
        # Normalize: 0,255 -> 0,1
        Xtrain = Xtrain / 255.0
        Xtest = Xtest / 255.0

        # step 1: find best hyperparameters using randomizedsearch
    # 1.1 set values of parameters for RF
    alpha = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
    param_grid_search = {"alpha": alpha}
    estimator = RidgeClassifier()
    # gsc = RandomizedSearchCV(estimator, param_distributions=param_grid_search, cv=cv, n_iter=5, return_train_score=False)
    gsc = GridSearchCV(estimator, param_grid_search, cv=cv, return_train_score=False)
    grid_result = gsc.fit(Xtrain, ytrain)
    grid_result_df = pd.DataFrame(grid_result.cv_results_)

    # 1.2 make plots of grid search results
    params_to_plot = ['param_alpha']
    cv_iter_num = np.arange(1, cv + 1)

    for p in params_to_plot:
        plt.figure(figsize=(3, 3))
        plt.plot(grid_result_df.loc[:, [p, "mean_test_score"]].groupby(p).mean(), marker="o")
        plt.ylabel("mean cv score")
        plt.grid(linestyle="--", alpha=0.5)
        plt.ylim(0.5, 1)
        plt.xlabel(p)
        # plt.title(f'Best mean cv score: {grid_result.best_score_} \n using {grid_result.best_params_}')
        plt.tight_layout()

        if normalized:
            plt.savefig(os.path.join(outpath, "RC_gridsearch__NORMALIZED" + p + ".png"), dpi=300)
        else:
            plt.savefig(os.path.join(outpath, "RC_gridsearch__" + p + ".png"), dpi=300)

    plt.close()

    # plot all test scores
    plt.figure(figsize=(3, 3))
    plt.plot(grid_result_df.loc[:, ['mean_test_score']].sum(axis=1).sort_values().values, marker=".",
             linestyle="")
    plt.ylabel("mean cv score")
    plt.grid(b=True, linestyle="--", alpha=0.5)
    plt.xlabel("parameter combination")
    plt.ylim(0.5, 1)
    plt.title('RC')
    plt.tight_layout()
    if normalized:
        plt.savefig(os.path.join(outpath, "RC_test_score__NORMALIZED.png"), dpi=300)
    else:
        plt.savefig(os.path.join(outpath, "RC_test_score.png"), dpi=300)
    plt.close()
    
    # run model with 9 random combinations of parameters
    random_combinations = grid_result_df["params"].sample(3)
    plt.figure(figsize=(3, 3))
    tmp_scores1 = []
    for i in random_combinations:
        tmp_model = RidgeClassifier(alpha=i["alpha"])
        model = tmp_model.fit(Xtrain,ytrain)
        tmp_scores1.append(accuracy_score(model.predict(Xtest), ytest))
        tmp_scores = cross_val_score(tmp_model, Xtest, ytest, cv=5)
        plt.plot(np.sort(tmp_scores), marker=".", linestyle=":", color="grey", alpha=.75)

    # run model with the best combination of parameters (from gridsearch)
    best_params = grid_result.best_params_
    
    best_mod = RidgeClassifier(alpha=best_params["alpha"])

    scores_test = cross_val_score(best_mod, Xtest, ytest, cv=5)

    plt.plot(np.sort(scores_test), marker="o", linestyle=":", color="red", alpha=.75,
             label="best combination")
    plt.ylabel("validation score")
    plt.grid(b=True, linestyle="--", alpha=0.5)
    plt.ylim(0.5, 1)
    plt.xlabel("CV iteration")
    plt.title("RC")
    plt.legend(loc = 'best')
    plt.tight_layout()
    if normalized:
        plt.savefig(os.path.join(outpath, "RC_test_cv__NORMALIZED.png"), dpi=300)
    else:
        plt.savefig(os.path.join(outpath, "RC_test_cv.png"), dpi=300)

    plt.close()
    # extracting best model
    best_model = grid_result.best_estimator_
    tmp_scores1.append(best_model.score(Xtest, ytest))

    score = calculate_acc_prec_rec_f1(best_model, Xtest, ytest)

    return best_model, score, grid_result, tmp_scores1


def classify_LR(Xtrain, ytrain, Xtest, ytest, normalized=False):
    if normalized:
        # Normalize: 0,255 -> 0,1
        Xtrain = Xtrain / 255.0
        Xtest = Xtest / 255.0

        # step 1: find best hyperparameters using randomizedsearch
    # 1.1 set values of parameters for RF
    param_grid_search = {"solver": ['newton-cg', 'liblinear'],
                         "C": [0.5, 1.0, 1.5]}
    estimator = LogisticRegression()
    # gsc = RandomizedSearchCV(estimator, param_distributions=param_grid_search, cv=cv, n_iter=5, return_train_score=False)
    gsc = GridSearchCV(estimator, param_grid_search, cv=cv, return_train_score=False)
    grid_result = gsc.fit(Xtrain, ytrain)
    grid_result_df = pd.DataFrame(grid_result.cv_results_)

    # 1.2 make plots of grid search results
    params_to_plot = ['param_solver', 'param_C']

    cv_iter_num = np.arange(1, cv + 1)
    for p in params_to_plot:
        plt.figure(figsize=(3, 3))
        plt.plot(grid_result_df.loc[:, [p, "mean_test_score"]].groupby(p).mean(), marker="o")
        plt.ylabel("mean cv score")
        plt.grid(linestyle="--", alpha=0.5)
        plt.xlabel(p)
        plt.ylim(0.5, 1)
        # plt.title(f'Best mean cv score: {grid_result.best_score_} \n using {grid_result.best_params_}')
        plt.tight_layout()

        if normalized:
            plt.savefig(os.path.join(outpath, "LR_gridsearch__NORMALIZED" + p + ".png"), dpi=300)
        else:
            plt.savefig(os.path.join(outpath, "LR_gridsearch__" + p + ".png"), dpi=300)

    plt.close()

    # plot all test scores
    plt.figure(figsize=(3, 3))
    plt.plot(grid_result_df.loc[:, ['mean_test_score']].sum(axis=1).sort_values().values, marker=".",
             linestyle="")
    plt.ylabel("mean cv score")
    plt.grid(b=True, linestyle="--", alpha=0.5)
    plt.xlabel("parameter combination")
    plt.title('LR')
    plt.ylim(0.5, 1)
    plt.tight_layout()
    if normalized:
        plt.savefig(os.path.join(outpath, "LR_test_score__NORMALIZED.png"), dpi=300)
    else:
        plt.savefig(os.path.join(outpath, "LR_test_score.png"), dpi=300)
        
    # run model with 9 random combinations of parameters
    random_combinations = grid_result_df["params"].sample(3)
    plt.figure(figsize=(3, 3))
    tmp_scores1 = []
    for i in random_combinations:
        tmp_model = LogisticRegression(solver=i["solver"], C= i['C'])
        model = tmp_model.fit(Xtrain,ytrain)
        tmp_scores1.append(accuracy_score(model.predict(Xtest), ytest))
        tmp_scores = cross_val_score(tmp_model, Xtest, ytest, cv=5)
        plt.plot(np.sort(tmp_scores), marker=".", linestyle=":", color="grey", alpha=.75)

    # run model with the best combination of parameters (from gridsearch)
    best_params = grid_result.best_params_
    
    best_mod = LogisticRegression(solver=best_params["solver"], C=best_params['C'])

    scores_test = cross_val_score(best_mod, Xtest, ytest, cv=5)

    plt.plot(np.sort(scores_test), marker="o", linestyle=":", color="red", alpha=.75,
             label="best combination")
    plt.ylabel("validation score")
    plt.grid(b=True, linestyle="--", alpha=0.5)
    plt.xlabel("CV iteration")
    plt.title("LR")
    plt.ylim(0.5, 1)
    plt.legend(loc = 'best')
    plt.tight_layout()
    if normalized:
        plt.savefig(os.path.join(outpath, "LR_test_cv__NORMALIZED.png"), dpi=300)
    else:
        plt.savefig(os.path.join(outpath, "LR_test_cv.png"), dpi=300)

    plt.close()

    # extracting best model
    best_model = grid_result.best_estimator_
    tmp_scores1.append(best_model.score(Xtest, ytest))

    score = calculate_acc_prec_rec_f1(best_model, Xtest, ytest)

    return best_model, score, grid_result, tmp_scores1


def classify_SVC(Xtrain, ytrain, Xtest, ytest, normalized=False):
    if normalized:
        # Normalize: 0,255 -> 0,1
        Xtrain = Xtrain / 255.0
        Xtest = Xtest / 255.0

    # step 1: find best hyperparameters using randomizedsearch
    # 1.1 set values of parameters for RF
    param_grid_search = {"kernel": ['poly', 'rbf', 'sigmoid'],
                         "gamma": ['scale'],
                         "C": [10, 1.0, 0.1, 0.01]}
    estimator = SVC()
    
    # gsc = RandomizedSearchCV(estimator, param_distributions=param_grid_search, cv=cv, n_iter=5, return_train_score=False)
    gsc = GridSearchCV(estimator, param_grid_search, cv=cv, return_train_score=False)
    grid_result = gsc.fit(Xtrain, ytrain)
    grid_result_df = pd.DataFrame(grid_result.cv_results_)

    # 1.2 make plots of grid search results
    params_to_plot = ['param_kernel', 'param_gamma', 'param_C']

    cv_iter_num = np.arange(1, cv + 1)
    for p in params_to_plot:
        plt.figure(figsize=(3, 3))
        plt.plot(grid_result_df.loc[:, [p, "mean_test_score"]].groupby(p).mean(), marker="o")
        plt.ylabel("mean cv score")
        plt.grid(linestyle="--", alpha=0.5)
        plt.ylim(0.5, 1)
        plt.xlabel(p)
        # plt.title(f'Best mean cv score: {grid_result.best_score_} \n using {grid_result.best_params_}')
        plt.tight_layout()

        if normalized:
            plt.savefig(os.path.join(outpath, "SVC_gridsearch__NORMALIZED" + p + ".png"), dpi=300)
        else:
            plt.savefig(os.path.join(outpath, "SVC_gridsearch__" + p + ".png"), dpi=300)

    plt.close()

    # plot all test scores
    plt.figure(figsize=(3, 3))
    plt.plot(grid_result_df.loc[:, ['mean_test_score']].sum(axis=1).sort_values().values, marker=".",
             linestyle="")
    plt.ylabel("mean cv score")
    plt.grid(b=True, linestyle="--", alpha=0.5)
    plt.xlabel("parameter combination")
    plt.title('SVC')
    plt.ylim(0.5, 1)
    plt.legend(loc='best')
    plt.tight_layout()
    if normalized:
        plt.savefig(os.path.join(outpath, "SVC_test_score__NORMALIZED.png"), dpi=300)
    else:
        plt.savefig(os.path.join(outpath, "SVC_test_score.png"), dpi=300)
        
        
    # run model with 9 random combinations of parameters
    random_combinations = grid_result_df["params"].sample(3)
    tmp_scores1 = []
    plt.figure(figsize=(3, 3))
    for i in random_combinations:
        tmp_model = SVC(kernel=i["kernel"], C = i['C'], gamma = i['gamma'])
        model = tmp_model.fit(Xtrain,ytrain)
        tmp_scores1.append(accuracy_score(model.predict(Xtest), ytest))
        tmp_scores = cross_val_score(tmp_model, Xtest, ytest, cv=5)
        plt.plot(np.sort(tmp_scores), marker=".", linestyle=":", color="grey", alpha=.75)

    # run model with the best combination of parameters (from gridsearch)
    best_params = grid_result.best_params_
    
    best_mod = SVC(kernel=best_params["kernel"], C = best_params['C'], gamma = best_params['gamma'])

    scores_test = cross_val_score(best_mod, Xtest, ytest, cv=5)

    plt.plot(np.sort(scores_test), marker="o", linestyle=":", color="red", alpha=.75,
             label="best combination")

    plt.ylabel("validation score")
    plt.grid(b=True, linestyle="--", alpha=0.5)
    plt.xlabel("CV iteration")
    plt.title("SVC")
    plt.ylim(0.5, 1)
    plt.tight_layout()
    if normalized:
        plt.savefig(os.path.join(outpath, "SVC_test_cv__NORMALIZED.png"), dpi=300)
    else:
        plt.savefig(os.path.join(outpath, "SVC_test_cv.png"), dpi=300)

    plt.close()


    # extracting best model
    best_model = grid_result.best_estimator_
    tmp_scores1.append(best_model.score(Xtest, ytest))

    score = calculate_acc_prec_rec_f1(best_model, Xtest, ytest)

    return best_model, score, grid_result, tmp_scores1


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num of GPUs Available:", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    data_path = r"dataset_npy"
    outpath = r"result"
    cv = 3

    # load data and labels
    color_images = np.load(os.path.join(data_path, 'data_images.npy'))
    labels = np.load(os.path.join(data_path, 'data_labels.npy'))

    # convert RGB images to gray images
    gray_images = []
    for color_image in color_images:
        gray_image = (cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY))
        gray_images.append(gray_image)

    gray_images = np.array(gray_images)

    # flatten the images
    flat_gray_images = gray_images.reshape(*gray_images.shape[:-2], -1)

    # split the data and its labels for training and testing purpose
    X_train, X_test, y_train, y_test = train_test_split(flat_gray_images, labels, test_size=0.3, shuffle=True, random_state=42)

    # ----------------------------------- RC ----------------------------------- #
    print('running ridge classifier...')
    model_RC, score_RC, grid_result_RC, rand_scores_RC = classify_RC(X_train, y_train, X_test, y_test)
    # save the model to disk
    filename = 'model_RC.sav'
    pickle.dump(model_RC, open(os.path.join('saved_model', filename), 'wb'))

    # # ----------------------------------- LR ----------------------------------- #
    print('running logistic regression...')
    model_LR, score_LR, grid_result_LR, rand_scores_LR = classify_LR(X_train, y_train, X_test, y_test)
    # save the model to disk
    filename = 'model_LR.sav'
    pickle.dump(model_LR, open(os.path.join('saved_model', filename), 'wb'))

    # # ----------------------------------- SVC ----------------------------------- #
    print('running SVC...')
    model_SVC, score_SVC, grid_result_SVC, rand_scores_SVC = classify_SVC(X_train, y_train, X_test, y_test)
    # save the model to disk
    filename = 'model_SVC.sav'
    pickle.dump(model_SVC, open(os.path.join('saved_model', filename), 'wb'))

    # ------------------------After Normalization ------------------------------- #
    print('--------------------  Normalizing input data   ---------------------')
    # ----------------------------------- RC ----------------------------------- #
    print('running ridge classifier...')
    model_RC_norm, score_RC_norm, grid_result_RC_norm, rand_scores_RC_norm = classify_RC(X_train, y_train, X_test, y_test, normalized=True)
    # save the model to disk
    filename = 'model_RC_norm.sav'
    pickle.dump(model_RC_norm, open(os.path.join('saved_model', filename), 'wb'))

    # # ----------------------------------- LR ----------------------------------- #
    print('running logistic regression...')
    model_LR_norm, score_LR_norm, grid_result_LR_norm, rand_scores_LR_norm = classify_LR(X_train, y_train, X_test, y_test, normalized=True)
    # save the model to disk
    filename = 'model_LR_norm.sav'
    pickle.dump(model_LR_norm, open(os.path.join('saved_model', filename), 'wb'))

    # # ----------------------------------- SVC ----------------------------------- #
    print('running SVC...')
    model_SVC_norm, score_SVC_norm, grid_result_SVC_norm, rand_scores_SVC_norm = classify_SVC(X_train, y_train, X_test, y_test, normalized=True)
    # save the model to disk
    filename = 'model_SVC_norm.sav'
    pickle.dump(model_SVC_norm, open(os.path.join('saved_model', filename), 'wb'))

    plt.figure(figsize=(3, 3))
    plt.plot(0, rand_scores_RC[3], marker="o", linestyle="", color="darkred", alpha=.75, label="best param comb")
    plt.plot(0, rand_scores_RC[0], marker="x", color="grey", alpha=.75, label="rand param comb 1")
    plt.plot(0, rand_scores_RC[1], marker="x", color="lightblue", alpha=.75,label="rand param comb 2")
    plt.plot(0, rand_scores_RC[2], marker="x", color="lightgreen", alpha=.75, label="rand param comb 1")
    plt.plot(1, rand_scores_LR[3], marker="o", color="darkred", alpha=.75)
    plt.plot(1, rand_scores_LR[0], marker="x", color="grey", alpha=.75)
    plt.plot(1, rand_scores_LR[1], marker="x", color="lightblue", alpha=.75)
    plt.plot(1, rand_scores_LR[2], marker="x", color="lightgreen", alpha=.75)
    plt.plot(2, rand_scores_SVC[3], marker="o", color="darkred", alpha=.75)
    plt.plot(2, rand_scores_SVC[0], marker="x", color="grey", alpha=.75)
    plt.plot(2, rand_scores_SVC[1], marker="x", color="lightblue", alpha=.75)
    plt.plot(2, rand_scores_SVC[2], marker="x", color="lightgreen", alpha=.75)
    plt.ylim(0.5, 1)
    plt.ylabel("accuracy")
    plt.grid(linestyle="--", alpha=0.5)
    plt.xlabel("classifiers")
    plt.xticks([0, 1, 2], ["RC", "LR", "SVC"])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, "SUMMARY_test_Acc_parameter.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=(3, 3))
    plt.plot(0, rand_scores_RC_norm[3], marker="o", linestyle="", color="darkred", alpha=.75, label="best param comb")
    plt.plot(0, rand_scores_RC_norm[0], marker="x", color="grey", alpha=.75, label="rand param comb 1")
    plt.plot(0, rand_scores_RC_norm[1], marker="x", color="lightblue", alpha=.75,label="rand param comb 2")
    plt.plot(0, rand_scores_RC_norm[2], marker="x", color="lightgreen", alpha=.75, label="rand param comb 1")
    plt.plot(1, rand_scores_LR_norm[3], marker="o", color="darkred", alpha=.75)
    plt.plot(1, rand_scores_LR_norm[0], marker="x", color="grey", alpha=.75)
    plt.plot(1, rand_scores_LR_norm[1], marker="x", color="lightblue", alpha=.75)
    plt.plot(1, rand_scores_LR_norm[2], marker="x", color="lightgreen", alpha=.75)
    plt.plot(2, rand_scores_SVC_norm[3], marker="o", color="darkred", alpha=.75)
    plt.plot(2, rand_scores_SVC_norm[0], marker="x", color="grey", alpha=.75)
    plt.plot(2, rand_scores_SVC_norm[1], marker="x", color="lightblue", alpha=.75)
    plt.plot(2, rand_scores_SVC_norm[2], marker="x", color="lightgreen", alpha=.75)
    plt.ylim(0.5, 1)
    plt.ylabel("accuracy")
    plt.grid(linestyle="--", alpha=0.5)
    plt.xlabel("classifiers")
    plt.xticks([0, 1, 2], ["RC", "LR", "SVC"])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, "SUMMARY_test_Acc_parameter__NORMALIZED.png"), dpi=300)
    plt.close()


    
    plt.figure(figsize=(3, 3))
    plt.plot(0, score_RC[0], marker="o", linestyle="", color="darkorange", alpha=.75, label="N")
    plt.plot(1, score_LR[0], marker="o", color="darkorange", alpha=.75)
    plt.plot(2, score_SVC[0], marker="o", color="darkorange", alpha=.75)
    plt.plot(0, score_RC_norm[0], marker="x", linestyle="", color="green", alpha=.75, label="Y")
    plt.plot(1, score_LR_norm[0], marker="x", color="green", alpha=.75)
    plt.plot(2, score_SVC_norm[0], marker="x", color="green", alpha=.75)
    plt.ylabel("accuracy")
    plt.ylim(0.5, 1)
    plt.grid(b=True, linestyle="--", alpha=0.5)
    plt.xlabel("classifier")
    plt.xticks([0, 1, 2], ["RC", "LR", "SVC"])
    plt.legend(title="scaled")
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, "SUMMARY_accuracy.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=(3, 3))
    plt.plot(0, score_RC[1], marker="o", linestyle="", color="darkorange", alpha=.75, label="N")
    plt.plot(1, score_LR[1], marker="o", color="darkorange", alpha=.75)
    plt.plot(2, score_SVC[1], marker="o", color="darkorange", alpha=.75)
    plt.plot(0, score_RC_norm[1], marker="x", linestyle="", color="green", alpha=.75, label="Y")
    plt.plot(1, score_LR_norm[1], marker="x", color="green", alpha=.75)
    plt.plot(2, score_SVC_norm[1], marker="x", color="green", alpha=.75)
    plt.ylabel("precision")
    plt.ylim(0.5, 1)
    plt.grid(b=True, linestyle="--", alpha=0.5)
    plt.xlabel("classifier")
    plt.xticks([0, 1, 2], ["RC", "LR", "SVC"])
    plt.legend(title="scaled")
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, "SUMMARY_precision.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=(3, 3))
    plt.plot(0, score_RC[2], marker="o", linestyle="", color="darkorange", alpha=.75, label="N")
    plt.plot(1, score_LR[2], marker="o", color="darkorange", alpha=.75)
    plt.plot(2, score_SVC[2], marker="o", color="darkorange", alpha=.75)
    plt.plot(0, score_RC_norm[2], marker="x", linestyle="", color="green", alpha=.75, label="Y")
    plt.plot(1, score_LR_norm[2], marker="x", color="green", alpha=.75)
    plt.plot(2, score_SVC_norm[2], marker="x", color="green", alpha=.75)
    plt.ylabel("f1")
    plt.ylim(0.5, 1)
    plt.grid(b=True, linestyle="--", alpha=0.5)
    plt.xlabel("classifier")
    plt.xticks([0, 1, 2], ["RC", "LR", "SVC"])
    plt.legend(title="scaled")
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, "SUMMARY_f1.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=(3, 3))
    plt.plot(0, score_RC[3], marker="o", linestyle="", color="darkorange", alpha=.75, label="N")
    plt.plot(1, score_LR[3], marker="o", color="darkorange", alpha=.75)
    plt.plot(2, score_SVC[3], marker="o", color="darkorange", alpha=.75)
    plt.plot(0, score_RC_norm[3], marker="x", linestyle="", color="green", alpha=.75, label="Y")
    plt.plot(1, score_LR_norm[3], marker="x", color="green", alpha=.75)
    plt.plot(2, score_SVC_norm[3], marker="x", color="green", alpha=.75)
    plt.ylabel("recall")
    plt.ylim(0.5, 1)
    plt.grid(b=True, linestyle="--", alpha=0.5)
    plt.xlabel("classifier")
    plt.xticks([0, 1, 2], ["RC", "LR", "SVC"])
    plt.legend(title="scaled")
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, "SUMMARY_recall.png"), dpi=300)
    plt.close()
    
    
