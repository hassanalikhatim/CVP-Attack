import numpy as np



class Dataset:
    
    def __init__(
        self,
        data_name, 
        preferred_size=None,
        max_len=-1
    ):
        
        self.data_name = data_name
        self.preferred_size = preferred_size
        self.max_len = max_len
        
        self.type = 'cv'
        
        return
    
    
    def renew_data(
        self,
        preferred_size=None
    ):
        
        if preferred_size is not None:
            self.preferred_size = preferred_size
        
        self.x_train, self.y_train, self.x_test, self.y_test = self.prepare_data()
        
        return
    
    
    def get_input_shape(self):
        return self.x_train.shape[1:]
    
    
    def get_output_shape(self):
        return self.y_train.shape[1:]
    
    
    def invert_color_scale(self):
        
        self.x_train = 1 - self.x_train
        self.x_test = 1 - self.x_test
        
        return
