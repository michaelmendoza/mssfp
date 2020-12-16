''' Deep SSFP '''

import os

import time
import numpy as np 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate
import matplotlib
import matplotlib.pyplot as plt

from phantom import generate_brain_dataset

class DataGenerator:

    def __init__(self, 
        data,
        width = 128, #196, 
        height = 128, #196, 
        ratio = 0.8, 
        useNormalization = False, 
        useWhitening = True, 
        useRandomOrder = False):
        
        self.WIDTH = width
        self.HEIGHT = height
        self.CHANNELS = 1
        self.ratio = ratio
        
        self.useNormalization = useNormalization
        self.useWhitening = useWhitening
        self.useRandomOrder = useRandomOrder

        print("Loading and formating image data ....")
        self.generate(data)
        print("Loading and formating image data: Complete")
        print("Data size: Input Data", self.x_train.shape, " Truth Data:", self.y_train.shape);

    def generate(self, data):
        self.data = data

        # Randomize data order
        if self.useRandomOrder:
            indices = [_ for _ in range(len(self.data))]
            self.data = self.data[indices]

        # Data preprocessing
        if self.useNormalization:
            self.data, self.img_min, self.img_max = self.normalize(self.data)

        if self.useWhitening:
            self.data, self.img_mean, self.img_std = self.whiten(self.data)

        # Format data into x, y vectors
        self.format_data()

        # Split data into test/training sets 
        index = int(self.ratio * len(self.x_data)) # Split index
        self.x_train = self.x_data[0:index, :]
        self.x_test = self.x_data[index:, :] 
        self.y_train = self.y_data[0:index]
        self.y_test = self.y_data[index:]

    def format_data(self):
        # (B, H, W, C) -> (B,H,W,C/2) 
        data = self.format_data_split_complex(self.data)
        self.x_data = data[:,:,:,0::2]
        self.y_data = data[:,:,:,1::2]

    def format_data_split_complex(self, data):
        s = data.shape
        _data = np.zeros((s[0], s[1], s[2], 2*s[3]))
        for n in range(s[3]):
            _data[:,:,:,2*n] = data[:,:,:,n].real
            _data[:,:,:,2*n+1] = data[:,:,:,n].imag
        return _data

    def next_batch(self, batch_size):
        length = self.x_train.shape[0]
        indices = np.random.randint(0, length, batch_size) # Grab batch_size values randomly
        return [self.x_train[indices], self.y_train[indices]]

    def normalize(self, data):
        max = np.max(data)
        min = np.min(data)
        return (data - min) / (max - min), min, max

    def whiten(self, data):
        mean = np.mean(data)
        std = np.std(data)
        print("mean: " + str(mean) + " std: " + str(std))
        return (data - mean) / std, mean, std

    def denormalize(self, data, min, max):
        return data * (max - min) + min

    def undo_whitening(self, data, mean, std):
        return data * std + mean

    def undo_whitening_and_denormalize(self, data, stats):
        return (data * stats.std + stats.mean) * (stats.max - stats.min) + stats.min

    def plot(self, index):
        plt.imshow(self.image_data[index,:,:,0])
        plt.show()

    def plot_formatted_data(self, input, output, results):
        imgs = []
        for n in range(4):
            imgs.append(input[:,:,2*n] + 1j * input[:,:,2*n+1])

        for n in range(4):
            imgs.append(output[:,:,2*n] + 1j * output[:,:,2*n+1])

        for n in range(4):
            imgs.append(results[:,:,2*n] + 1j * results[:,:,2*n+1])

        nx, ny = 3, 4
        plt.figure()
        for ii in range(nx*ny):
            plt.subplot(nx, ny, ii+1)
            plt.imshow(np.abs(imgs[ii]))
        plt.show() 


# Simple Model Architecture
def simple_model(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS):
    xin = keras.Input(shape=(HEIGHT, WIDTH, CHANNELS), name='img')
    x = Conv2D(32, (3, 3), padding="same", activation=tf.nn.relu)(xin)
    x = Conv2D(32, (3, 3), padding="same", activation=tf.nn.relu)(x)
    xout = Conv2D(NUM_OUTPUTS, (1, 1), padding="same", activation=tf.nn.softmax)(x)
    return tf.keras.Model(inputs=xin, outputs=xout)

# Unet Model Architecture
def unet_model(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS):

    def down_block(x, filters):
        x = Conv2D(filters, (3, 3), padding="same", activation=tf.nn.relu, kernel_initializer='he_normal')(x)
        x = Conv2D(filters, (3, 3), padding="same", activation=tf.nn.relu, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=-1, momentum=0.95, epsilon=0.001)(x)
        #x = Dropout(rate=0.0)(x)
        return x

    def max_pool(x):
        return MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2))(x)

    def up_block(x, filters, skip_connect):
        x = Conv2DTranspose(filters, (3, 3), strides=2, padding="same", activation=tf.nn.relu)(x)
        x = concatenate([x, skip_connect], axis=3)
        x = Conv2D(filters, (3, 3), padding="same", activation=tf.nn.relu, kernel_initializer='he_normal')(x)
        x = Conv2D(filters, (3, 3), padding="same", activation=tf.nn.relu, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=-1, momentum=0.95, epsilon=0.001)(x)
        #x = Dropout(rate=0.0)(x)
        return x 

    def unet():
        fn = [32, 64, 128, 256, 512]
        fdepth = len(fn)

        x_stack = []
        xin = keras.Input(shape=(HEIGHT, WIDTH, CHANNELS), name='img')

        x = xin
        for idx in range(fdepth):
            x = down_block(x, fn[idx])

            if(idx < fdepth - 1):
                x_stack.append(x)
                x = max_pool(x)

        for idx in range(fdepth - 1):
            idx = fdepth - idx - 2
            x = up_block(x, fn[idx], x_stack.pop())

        xout = Conv2D(NUM_OUTPUTS, (1, 1), padding="same", activation=tf.nn.softmax)(x)
        return tf.keras.Model(inputs=xin, outputs=xout)

    return unet()

def train_model():

    # Training Parameters
    epochs = 100
    batch_size = 16 
    test_batch_size = 8

    #dataset = generate_brain_dataset()
    dataset = np.load('./data/phantom_data.npy')
    data = DataGenerator(dataset, width=128, height=128)
    
    x_train = data.x_train
    y_train = data.y_train
    x_test = data.x_test
    y_test = data.y_test
    print("Training DataSet: " + str(x_train.shape) + " " + str(y_train.shape))
    print("Test DataSet: " + str(x_test.shape) + " " + str(y_test.shape))

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).shuffle(1000)
    train_dataset = train_dataset.repeat()

    valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(test_batch_size).shuffle(1000)
    valid_dataset = valid_dataset.repeat()

    # Network Parameters
    WIDTH = data.WIDTH
    HEIGHT = data.HEIGHT
    CHANNELS = 8
    NUM_OUTPUTS = 8

    model = unet_model(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS)
    #model = simple_model(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS)

    model.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['categorical_accuracy'])
    model.summary()

    start = time.time()
    history = model.fit(train_dataset, 
            epochs=epochs, 
            steps_per_epoch=20,
            validation_data=valid_dataset,
            validation_steps = 10)

    evaluation = model.evaluate(x_test, y_test, verbose=1)
    predictions = model.predict(data.x_test)
    end = time.time()

    print('Summary: Accuracy: %.2f Time Elapsed: %.2f seconds' % (evaluation[1], (end - start)) )

    index = 0
    data.plot_formatted_data(data.x_test[index], data.y_test[index], predictions[index])
    
    # Save model 
    print("Save model")
    model.save("synethetic_pc_model")

def load_model():
    dataset = generate_brain_dataset()
    data = DataGenerator(dataset, width=128, height=128)
    x_train = data.x_train
    y_train = data.y_train
    x_test = data.x_test
    y_test = data.y_test

    model = keras.models.load_model("data/synethetic_pc_model")
    index = 100
    predictions = model.predict(data.x_test)
    data.plot_formatted_data(data.x_test[index], data.y_test[index], predictions[index])

def generate_data_set():
    dataset = generate_brain_dataset()
    print(dataset.shape)
    np.save('./data/phantom_data.npy', dataset)

train_model()
#load_model()
#generate_data_set()