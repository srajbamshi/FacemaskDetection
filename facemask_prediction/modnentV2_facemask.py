import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from sklearn.model_selection import train_test_split
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
    
    # Normalize: 0,255 -> 0,1
    Xtrain, Xtest = Xtrain / 255.0, Xtest / 255.0
    
    # instantiates the MobileNetV2 architecture without the last layer
    # 1) instantiates the MobileNetV2 architecture
    mobile = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(96, 96, 3), weights='imagenet')

    #drop last layer
    mobile = tf.keras.Model(inputs=mobile.input, outputs=mobile.layers[-2].output)
    mobile.summary()
    # make loaded layers non-trainable
    for layer in mobile.layers:
        layer.trainable = False

    # add new output layer customized to the dataset
    model = tf.keras.Sequential()
    model.add(mobile)
    model.add(tf.keras.layers.Dense(2,activation='softmax'))

    # print summary
    model.summary()

    # compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=["accuracy"])

    # training
    batch_size = 64
    epochs = 10
    
    print(f'initial training loss, initial training accuracy: {model.evaluate(Xtrain,  ytrain, batch_size=batch_size, verbose=2)}')
    
    history = model.fit(Xtrain, ytrain, validation_split=0.3, epochs=epochs,
              batch_size=batch_size, verbose=2)
    
    # evaluation on the test set
    test_loss, test_accuracy = model.evaluate(Xtest,  ytest, batch_size=batch_size, verbose=2)
    print(f'test loss, test accuracy: {model.evaluate(Xtest,  ytest, batch_size=batch_size, verbose=2)}')
    
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
    plt.grid('on')
    plt.tight_layout()
    
    plt.savefig(os.path.join('result', "Fig 2.3 model_loss_accuracy.png"), dpi=fig.dpi)
    plt.show()
    
    # Save the entire model as a SavedModel.
    model.save(os.path.join('saved_model', 'model_2.3'))
    
    from tensorflow.keras.models import load_model
    model1 = load_model(os.path.join('saved_model', 'model_2.3'))
    print("Base accuracy on original images:", model1.evaluate(x=Xtest, y=ytest, verbose=0))
    #np.testing.assert_allclose(model.predict(Xtest), model1.predict(Xtest))
    

