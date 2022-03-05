import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_images(path):
    images = []
    for img in os.listdir(path):
        if img != ".DS_Store":
            imgPath = os.path.join(path, img)
            image = cv2.imread(imgPath)[..., ::-1]  # read and convert BGR to RGB
            image = cv2.resize(image, (96, 96))
            images.append(image)
    return np.array(images)

if __name__ == "__main__":
    image_dir = r"Materials-20211215/facemask_dataset/dataset"
    print(os.listdir(image_dir))
    for dir in os.listdir(image_dir):
        if dir != ".DS_Store":
            color_images = get_images(os.path.join(image_dir, dir))

            # instantiates the MobileNetV2 architecture
            mobile = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(96, 96, 3), weights='imagenet')

            # prepare images for mobilenetv2 (scaling)
            color_images = color_images / 255.0
            # color_images = np.array(tf.keras.applications.mobilenet.preprocess_input(color_images))

            # predict labels
            predicted_labels = mobile.predict(color_images)

            # decode predictions
            decoded_predicted_labels = np.array(
                tf.keras.applications.mobilenet_v2.decode_predictions(predicted_labels, top=1))

            # all predictions
            all_predicted_labels = []
            for pred in decoded_predicted_labels:
                b = np.transpose(pred)
                all_predicted_labels.append(b[1])

            all_predicted_labels = np.array(all_predicted_labels)
            #print(all_predicted_labels)

            # unique predictions
            unique_predicted_labels, counts = np.unique(all_predicted_labels, return_counts=True)

            df1 = pd.DataFrame()
            df1['labels'] = unique_predicted_labels
            df1['counts'] = counts
            df1 = df1[df1['counts'] > 10]
            df1 = df1.sort_values(by=['counts'], ascending=False)

            fig, ax = plt.subplots()
            ax.bar(df1['labels'], df1['counts'])
            ax.set_xticks(df1['labels'], df1['labels'], rotation=90, fontsize=7.5)
            plt.xlabel('classes')
            plt.ylabel('frequency')
            plt.tight_layout()
            plt.title('dataset ('+str(dir)+')')
            plt.savefig(os.path.join("result", "Fig 2.1 distribution_"+str(dir)+'.png'), dpi=300)
            plt.show()

