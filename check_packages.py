import tensorflow as tf
# print("GPU available:", tf.config.list_physical_devices())
print("No of GPU available:", len(tf.config.list_physical_devices()))
print("TF version:", tf.__version__)

import sklearn
print("sklearn version:", sklearn.__version__)

import matplotlib
print("matplotlib version:", matplotlib.__version__)

import pandas
print("pandas version:", pandas.__version__)

import numpy
print("numpy version:", numpy.__version__)

import cv2
print("cv2 version:", cv2.__version__)
