import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import keras
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


from keras.datasets import mnist
(mnist_x_train, mnist_labels_train), (mnist_x_test, mnist_labels_test) = mnist.load_data()

mnist_x_train = mnist_x_train.astype('float32')
mnist_x_test = mnist_x_test.astype('float32')
mnist_x_train /= 255
mnist_x_test /= 255
mnist_y_train = to_categorical(mnist_labels_train, 10)
mnist_y_test = to_categorical(mnist_labels_test, 10)
mnist_x_train = mnist_x_train.reshape(mnist_x_train.shape[0], 28, 28, 1)
mnist_x_test = mnist_x_test.reshape(mnist_x_test.shape[0], 28, 28, 1)

X_total = np.vstack((mnist_x_train,mnist_x_test))

Y_total = np.vstack((mnist_y_train,mnist_y_test))

# Set the random seed
random_seed = 4
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_total, Y_total, test_size = 0.1, random_state=random_seed)


def build_model(): 
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                    activation ='relu', input_shape = (28,28,1)))
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                    activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = "softmax"))
    return model

gen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


gen.fit(X_train)
batch_size = 86
train_generator = gen.flow(X_train, Y_train, batch_size=batch_size)

epochs = 30
net = build_model()
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
# Define the optimizer
optimizer = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
net.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# fits the model on batches with real-time data augmentation:
net.fit_generator(train_generator,epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])          

net.save("all_total_net.h5")

outputs=net.predict(mnist_x_test)
labels_predicted=np.argmax(outputs, axis=1)
misclassified=sum(labels_predicted==mnist_labels_test)
print('Percentage classified = ',100*misclassified/mnist_labels_test.size)
