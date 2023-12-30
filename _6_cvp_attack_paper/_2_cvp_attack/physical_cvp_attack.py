import tensorflow as tf
import numpy as np
import time


from _0_general_ML.data_utils.dataset_cards.taxibj import TaxiBJ
from _0_general_ML.model_utils.model import Keras_Model

from .cvp_attack import CVP_Attack



class CVP_Attack(CVP_Attack):
    
    def __init__(
        self, 
        data: TaxiBJ, model: Keras_Model, 
        input_mask=None, output_mask=None
    ):
        
        super().__init__(
            data, model, 
            input_mask, output_mask
        )
        
        return
    
    
    def attack(
        self, x_input, y_input, 
        device_budget=15000,
        iterations=1000,
        targeted=False, 
        **kwargs
    ):
        '''
        Inputs:
            {x_input}: the crowd-flow state input of shape (None, 1, 32, 32, 2*history_length)
            {y_input}: the ground-truth crowd-flow state of shape (None, 32, 32, 2)
            {device_budget}: the maximum allowed perturbation to a pixel
        '''
        
        epsilon = device_budget/1000
        
        history = self.data.data_configuration['history_length']
        self.last_run_loss_values = []
        previous_print_time = 0
        epsilon_per_iteration = epsilon/(iterations/4)
        
        # x_perturbation = -5*np.ones([history, 9]+list(x_input.shape[2:4])).astype(np.float32)
        # shape of x_perturbation: [1, 25, 32, 32]
        x_perturbation = -5*np.ones([1, 25]+list(x_input.shape[2:4])).astype(np.float32)
        x_perturbation[:,0] = 0.
        for iteration in range(iterations):
            
            # Just to monitor the progress
            if time.time() - previous_print_time > 1:
                previous_print_time = time.time()
                print("\rAttacking: ", iteration, "/", iterations, "\t", end="")
            
            p_perturbation = self.cvp_attack_step(
                x_input, y_input, x_perturbation,
                epsilon=epsilon_per_iteration,
                targeted=targeted
            )
            x_perturbation[:,:1] -= epsilon_per_iteration*p_perturbation[:,:1]
            x_perturbation[:,1:] -= (10/iterations)*p_perturbation[:,1:]
            
            # only positive perturbations
            x_perturbation[:,0] = np.clip(x_perturbation[:,0], 0, 1)
            
            # L1 norm based clipping
            current_l1_norm = np.expand_dims(
                np.mean(np.abs(x_perturbation[:,0]), axis=(1,2)), 
                axis=(0,2,3)
            )
            if np.max(current_l1_norm) > epsilon:
                indices = np.where( current_l1_norm > epsilon )
                x_perturbation[indices,0] *= epsilon / current_l1_norm[:,indices]
        
        x_perturbation_real = self.compute_inflow_outflow(x_perturbation)
        
        return np.clip(x_input+x_perturbation_real, 0, 1)
    
    