import tensorflow as tf
import numpy as np


from _0_general_ML.data_utils.dataset import Dataset
from _0_general_ML.model_utils.model import Keras_Model

from .adaptive_attack import Adaptive_Attack



class Adaptive_i_FGSM_Attack(Adaptive_Attack):
    
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
        iterations=100, 
        targeted=False, **kwargs
    ):
        
        self.last_run_loss_values = []
        epsilon_per_iteration = epsilon/iterations
        x_perturbation = np.zeros_like(x_input).astype(np.float32)

        for iteration in range(iterations):
            x_perturbation = self.fgsm_step(
                x_input, y_input, x_perturbation, 
                epsilon=epsilon_per_iteration,
                targeted=targeted
            )
            x_perturbation = np.clip(x_input+x_perturbation, 0., 1.) - x_input
            
            x_perturbation = self.universalize_perturbation(x_perturbation)
            
        return np.clip(x_input + x_perturbation, 0., 1.)
    
    