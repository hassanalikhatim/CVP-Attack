import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from utils_.general_utils import confirm_directory

from ..data_utils.dataset import Dataset

from .model_architectures.model_architectures import *



WEIGHT_DECAY = 1e-2

model_architectures = {
    'mlp': mlp_baseline,
    'tgcn': TGCN_baseline,
    'st_resnet': st_resnet_baseline
}


class Keras_Model:
    
    def __init__(
        self, 
        data: Dataset, model_configuration,
        path=None
    ):
        
        self.path = path
        self.data = data
        
        self.prepare_model(model_configuration)
        
        return
    
    
    def prepare_model(self, model_configuration):
        
        self.model_configuration = {
            'model_architecture': 'mlp',
            'learning_rate': 1e-4,
            'weight_decay': WEIGHT_DECAY,
            'parameters': None
        }
        if model_configuration:
            for key in model_configuration.keys():
                self.model_configuration[key] = model_configuration[key]
        
        self.model = model_architectures[self.model_configuration['model_architecture']](
            self.data,
            self.model_configuration
        )
        
        self.save_directory = self.path + self.data.data_name + '/'
        
        return
    
    
    def train(self, epochs=1, batch_size=None, patience=15):
        
        early_stopping_monitor = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=patience,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True)

        self.model.fit(
            self.data.x_train, self.data.y_train,
            epochs=epochs, batch_size=batch_size,
            validation_data=(self.data.x_test, self.data.y_test),
            callbacks=[early_stopping_monitor])
        
        return
        
        
    def save(self, name):
        
        confirm_directory(self.save_directory)
        self.model.save_weights(self.save_directory + name + '.h5')
        
        return
        
        
    def load_or_train(self, name, epochs=1, batch_size=None, patience=1):

        if os.path.isfile(self.save_directory + name + '.h5'):
            self.model.load_weights(self.save_directory + name + '.h5')
            print('Loaded pretrained weights:', name)
        
        else:
            print('Model not found at: ', self.save_directory + name + '.h5')
            print('Training model from scratch.')
            self.train(epochs=epochs, batch_size=batch_size, patience=patience)
            self.save(name)

        return
