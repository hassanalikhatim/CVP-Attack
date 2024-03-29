import tensorflow as tf
import numpy as np


from _0_general_ML.data_utils.dataset import Dataset
from _0_general_ML.model_utils.model import Keras_Model

from .adaptive_attack import Adaptive_Attack



class Physical_PGD_Attack(Adaptive_Attack):
    
    def __init__(
        self, 
        data: Dataset, model: Keras_Model, 
        input_mask=None, output_mask=None
    ):
        
        super().__init__(
            data, model, 
            input_mask, output_mask
        )
        
        self.name = 'Physical PGD Attack'
        self.device_budget = None
        
        return
    
    
    def attack(
        self, x_input, y_input,
        device_budget=15000,
        iterations=1000,
        targeted=False, 
        **kwargs
    ):
        
        self.device_budget = device_budget
        epsilon = device_budget/1000
        
        self.last_run_loss_values = []
        epsilon_per_iteration = epsilon/(iterations/4)

        x_perturbation = np.zeros_like(x_input).astype(np.float32)
        for iteration in range(iterations):
            x_perturbation = self.fgsm_step(
                x_input, y_input, x_perturbation, 
                epsilon=epsilon_per_iteration,
                targeted=targeted
            )
            
            # Only positive perturbations
            x_perturbation = np.clip(x_perturbation, 0, 1)
            
            # L1-Norm
            current_l1_norm = np.expand_dims(
                np.sum(np.abs(x_perturbation), axis=(1,2,3)), 
                axis=(1,2,3)
            )
            if np.max(current_l1_norm) > epsilon:
                indices = np.where( current_l1_norm > epsilon )
                x_perturbation[indices] *= epsilon / current_l1_norm[indices]
            
            x_perturbation = np.clip(x_input+x_perturbation, 0, 1) - x_input
            
            x_perturbation = self.universalize_perturbation(x_perturbation)
            
        return np.clip(x_input + x_perturbation, 0, 1)
    
    
    def info(self):
        print('\n\n')
        print('{} | {}'.format(self.name, self.device_budget))
        return
    
    