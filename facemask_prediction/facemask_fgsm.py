import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Function to create adversarial pattern and create adversary
def generate_adversary(image, label, model):
    image = tf.cast(image, tf.float32)
    label = tf.reshape(label, (1, 2))
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    # print(type(image))

    with tf.GradientTape() as tape:
        # explicitly indicate that our image should be tacked for
        # gradient updates
        tape.watch(image)
        # use our model to make predictions on the input image and
        # then compute the loss
        prediction = model(image)
        loss = loss_object(label, prediction)

    # calculate the gradients of loss with respect to the image
    gradient = tape.gradient(loss, image)

    # compute the sign of the gradient(adversarial pattern)
    signed_grad = tf.sign(gradient)

    adversarial_example = image + signed_grad * 0.1
    return adversarial_example.numpy()


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

    class_names = ['with_mask', 'without_mask']

    # load model
    model1 = load_model(os.path.join('saved_model', 'model_1.2_latest'))
    model1.summary()

    print("Base accuracy on original images:", model1.evaluate(x=Xtest, y=ytest, verbose=0))
    
    # ################### with mobnet
    # load model
    # instantiates the MobileNetV2 architecture without the last layer
    # 1) instantiates the MobileNetV2 architecture
    # load model
    model2 = load_model(os.path.join('saved_model', 'model_2.3'))
    model2.summary()

    print("Base accuracy on original images:", model2.evaluate(x=Xtest, y=ytest, verbose=0))
    
    # define subplot grid
    fig1, axs1 = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
    fig2, axs2 = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
    fig3, axs3 = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
    # loop through axes
    for ax1, ax2, ax3 in zip(axs1.ravel(),axs2.ravel(), axs3.ravel()):
        i = np.random.randint(0, len(Xtest))
        im = Xtest[i]
        image = np.expand_dims(im, axis=0)
        image_label = ytest[i]
        ax1.imshow((image.squeeze() * 255).astype(np.uint8))
        ax1.set_title(
            f'Ground Truth: {class_names[image_label.argmax()]} \n'
            f'Image Prediction:  {class_names[model1.predict(image).argmax()]}')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.grid(False)
        
        # model 1
        adversary = generate_adversary(image, image_label, model1)
        ax2.imshow((adversary.squeeze() * 255).astype(np.uint8))
        ax2.set_title(f'Adversarial Prediction: \n {class_names[model1.predict(adversary).argmax()]}')
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.grid(False)
        
        # model 2
        adversary = generate_adversary(image, image_label, model2)
        ax3.imshow((adversary.squeeze() * 255).astype(np.uint8))
        ax3.set_title(f'Adversarial Prediction:\n {class_names[model2.predict(adversary).argmax()]}')
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.grid(False)
    fig1.savefig(os.path.join('result', "Fig 3.3orig.png"), dpi=300)
    fig2.savefig(os.path.join('result', "Fig 3.3adv.png"), dpi=300)
    fig3.savefig(os.path.join('result', "Fig 3.3adv_mobnet.png"), dpi=300)
    

    
    
   