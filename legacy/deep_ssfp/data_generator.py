
import numpy as np
from deep_ssfp.format import format_evenodd 

class DataGenerator:

    def __init__(self, 
        data,
        width = 128, 
        height = 128, 
        ratio = 0.8, 
        useNormalization = False, 
        useWhitening = True, 
        useRandomOrder = False,
        datatype='evenodd'):
        
        self.WIDTH = width
        self.HEIGHT = height
        self.CHANNELS = 1
        self.ratio = ratio
        
        self.useNormalization = useNormalization
        self.useWhitening = useWhitening
        self.useRandomOrder = useRandomOrder
        self.datatype = datatype

        print("Loading and formating image data ....")
        self.generate(data)
        print("Loading and formating image data: Complete")
        print("Train data size: Input Data", self.x_train.shape, " Truth Data:", self.y_train.shape)
        print("Test data size: Input Data", self.x_test.shape, " Truth Data:", self.y_test.shape)

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
        x, y = self.format_data()

        # Split data into test/training sets 
        index = int(self.ratio * len(x)) # Split index
        self.x_train = x[0:index]
        self.x_test = x[index:] 
        self.y_train = y[0:index]
        self.y_test = y[index:]

    def format_data(self):
        # Formats data according to desired format 
        if self.datatype == 'evenodd':
            return format_evenodd(self.data)
        else:
            return np.array([]), np.array([])

    def normalize(self, data):
        # Min-Max Scaler 
        max = np.max(data)
        min = np.min(data)
        return (data - min) / (max - min), min, max

    def whiten(self, data):
        # Standard Scaler 
        mean = np.mean(data)
        std = np.std(data)
        print("mean: " + str(mean) + " std: " + str(std))
        return (data - mean) / std, mean, std

    def denormalize(self, data, min, max):
        return data * (max - min) + min

    def undo_whitening(self, data, mean, std):
        return data * std + mean
