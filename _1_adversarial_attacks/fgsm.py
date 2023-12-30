import tensorflow as tf
import numpy as np


from _0_general_ML.data_utils.dataset import Dataset
from _0_general_ML.model_utils.model import Keras_Model

from .attack import Attack



class FGSM_Attack(Attack):
    
    def __init__(
        self, 
        data: Dataset, model: Keras_Model, 
        input_mask=None, output_mask=None
    ):
        
        super().__init__(
            data, model, 
            input_mask, output_mask
        )
        
        return
    
    
    def attack(
        self, 
        x_input, y_input, 
        epsilon=0.1,
        targeted=False,
        **kwargs
    ):
        
        self.last_run_loss_values = []
        x_perturbation = np.zeros_like(x_input).astype(np.float32)
        
        x_perturbation = self.fgsm_step(
            x_input, y_input, x_perturbation, 
            epsilon=epsilon, targeted=targeted
        )
        
        return np.clip(x_input+x_perturbation, 0, 1)