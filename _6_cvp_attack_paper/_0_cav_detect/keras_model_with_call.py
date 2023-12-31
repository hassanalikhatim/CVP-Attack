from _0_general_ML.data_utils.dataset import Dataset
from _0_general_ML.model_utils.model import Keras_Model



class Keras_Model_with_call(Keras_Model):
    
    def __init__(self, data: Dataset, model_configuration, path=None):
        super().__init__(data, model_configuration, path)
        self.defended = False
        return
    
    
    def __call__(self, x_input):
        return self.model(x_input)