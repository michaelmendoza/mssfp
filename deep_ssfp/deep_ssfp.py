''' Deep SSFP '''

import os
from typing import Any
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import numpy as np 
import tensorflow as tf 
from tensorflow import keras

from plots import plot_formatted_data
from phantom import generate_ssfp_dataset
from .data_generator import DataGenerator
from .models import simple_model, unet_model

def train(dataset):

    # Training Parameters
    epochs = 100
    batch_size = 16 
    test_batch_size = 8

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
    
    # Save model 
    print("Save model")
    model.save("synethetic_phase_cycle-oddeven")

    index = 0
    plot_formatted_data(data.x_test[index], data.y_test[index], predictions[index])
    
    return model, history

def load_model():
    dataset : Any = load_dataset()
    data = DataGenerator(dataset.item()['M'], width=128, height=128)

    model : Any = keras.models.load_model("synethetic_phase_cycle-oddeven")
    index = 100
    predictions = model.predict(data.x_test)
    plot_formatted_data(data.x_test[index], data.y_test[index], predictions[index])

def generate_dataset_and_save():
    dataset = generate_ssfp_dataset()
    print(dataset['M'].shape)
    np.save('./phantom_brain_ssfp_data.npy', dataset)
    return dataset 

def load_dataset():
    return np.load('./phantom_brain_ssfp_data.npy', allow_pickle=True)

if __name__ == "__main__":
    #train()
    #load_model()
    #generate_data_set()
    pass