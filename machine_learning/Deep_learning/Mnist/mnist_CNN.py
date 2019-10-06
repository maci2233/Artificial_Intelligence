import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras import models
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

#benji's magic code so that our classmates don't hate us :)
def get_session(gpu_fraction=0.3):
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
  return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(get_session())


def pause():
	input('press Enter to continue...')

#Using pandas to read the train and test csv files.
train = pd.read_csv('./train.csv')

test = pd.read_csv('./test.csv')

#All the values that start from column 1 (zero based) to the end are pixels, we are going to change them to float type
#and normalize them dividing them /255
x_train = (train.ix[:,1:].values).astype('float32')/255.0

#Column 0 contains the labels so we are just going to turn all of them to integers
train_labels = train.ix[:,0].values.astype('int32')

#Same as the triain pixels but for the test pixels to make the predictions, this time we do it for all the columns
#since there are no labels.
x_test = test.values.astype('float32')/255.0

#So far every sample is in 1D, which is not the shape that keras expects, wre are going to change it to a 28x28x1 so that
#keras can make the convolutions, we do not need more than 2 Dimensions because the pixels contain grayscale values.
train_images = x_train.reshape(x_train.shape[0], 28, 28, 1)
test_images = x_test.reshape(x_test.shape[0], 28, 28, 1)

#Telling keras that we want 10 categories for the labels.
train_labels = keras.utils.to_categorical(train_labels, 10)

#This is a simple neural network that receives the samples will receive reshaped samples
#Relu is the most popular activation functions for this type of neural networks
#the kernel size is 3x3 and the pooling will be of 2x2
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
#We add a final dense layer that has the 10 different categories, we use softmax here because we need an activation function that can classify
#to any given number of classes (10 in this case)
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#We train the model with the train images and the train labels for 3 epochs
model.fit(train_images, train_labels, epochs=3, batch_size=64)

#once the model is done with the training we are going to make predictions for the test images
predicted_classes = model.predict_classes(test_images)

#Based on the predictions we can use pandas to generate a csv with the format that kaggles asks us for
submissions=pd.DataFrame({"ImageId": list(range(1,len(predicted_classes)+1)),
                         "Label": predicted_classes})

submissions.to_csv("results.csv", index=False, header=True)
