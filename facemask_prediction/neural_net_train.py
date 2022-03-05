import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, Dropout, Reshape, Add

import matplotlib.pyplot as plt

if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num of GPUs Available:", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    data_path = r"dataset_npy"
    outpath = r"result"

    # load data and labels
    color_images = np.load(os.path.join(data_path, 'data_images.npy'))
    labels = np.load(os.path.join(data_path, 'data_labels.npy'))

    # creating one hot encoded labels
    labels = tf.keras.utils.to_categorical(labels, 2)

    # split the data and its labels for training and testing purpose
    Xtrain, Xtest, ytrain, ytest = train_test_split(color_images, labels, test_size=0.3, shuffle=True, random_state=42)
    print(Xtrain.shape)  # (2683, 96, 96, 3)

    # Normalize: 0,255 -> 0,1
    Xtrain, Xtest = Xtrain / 255.0, Xtest / 255.0

    class_names = ['with_mask', 'without_mask']

    # model...
    # training
    batch_size = 64
    epochs = 21

    # model...
    inputs = Input(shape=(Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3]))
    net = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    net = Conv2D(32, kernel_size=(3, 3), activation='relu')(net)
    net = MaxPooling2D((2, 2))(net)

    net = Conv2D(64, kernel_size=(3, 3), activation='relu')(net)
    net = Conv2D(64, kernel_size=(3, 3), activation='relu')(net)
    net = MaxPooling2D((2, 2))(net)

    net = Conv2D(128, kernel_size=(3, 3), activation='relu')(net)
    net = Conv2D(128, kernel_size=(3, 3), activation='relu')(net)
    net = MaxPooling2D((2, 2))(net)

    net = Conv2D(256, kernel_size=(3, 3), activation='relu')(net)
    net = Conv2D(256, kernel_size=(3, 3), activation='relu')(net)
    net = MaxPooling2D((2, 2))(net)

    net = Flatten()(net)
    net = Dense(256, activation='relu')(net)
    net = Dropout(0.5)(net)

    net = Dense(256, activation='relu')(net)
    net = Dropout(0.5)(net)

    #outputs = Dense(1, activation='sigmoid')(net)
    outputs = Dense(2, activation='softmax')(net)

    model = Model(inputs=inputs, outputs=outputs, name='FacemaskDetector')
    print(model.summary())

    # # loss and optimizer

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

    print(
        f'initial training loss, initial training accuracy:'
        f' {model.evaluate(Xtrain, ytrain, batch_size=batch_size, verbose=2)}')

    history = model.fit(Xtrain, ytrain, validation_split=0.3, epochs=epochs,
                        batch_size=batch_size, verbose=2)

    # evaluation on the test set
    test_loss, test_accuracy = model.evaluate(Xtest, ytest, batch_size=batch_size, verbose=2)
    print(f'test loss, test accuracy: {model.evaluate(Xtest, ytest, batch_size=batch_size, verbose=2)}')

    # Plot results
    ep = np.array(range(epochs)) + 1
    fig = plt.figure(figsize=(10, 5))

    # summarize history for accuracy
    plt.subplot(1, 2, 1)
    plt.plot(ep, history.history['accuracy'])
    plt.plot(ep, history.history['val_accuracy'])
    plt.title(f'test accuracy = {test_accuracy * 100:.2f}%', fontsize=15)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    #plt.xticks(ep, ep)
    #plt.ylim([0, 1])
    plt.legend(['training', 'validation'], loc='best')
    plt.grid('on')
    plt.tight_layout()

    # summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(ep, history.history['loss'])
    plt.plot(ep, history.history['val_loss'])
    plt.title(f'test loss = {test_loss:.2f}', fontsize=15)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    #plt.xticks(ep, ep)
    #plt.ylim([0, 1])
    plt.grid('on')
    plt.tight_layout()
    plt.savefig(os.path.join('result', "Fig 1.2 model_loss_accuracy1.png"), dpi=fig.dpi)
    plt.show()

    # save model
    model.save(os.path.join('saved_model', 'model_1.2_latest'))


