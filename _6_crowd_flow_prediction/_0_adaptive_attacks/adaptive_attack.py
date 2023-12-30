import tensorflow as tf
import numpy as np


from _0_general_ML.data_utils.dataset import Dataset

from _0_general_ML.model_utils.model import Keras_Model

from _1_adversarial_attacks.attack import Attack



eps = 0


class Adaptive_Attack(Attack):
    
    def __init__(
        self, 
        data: Dataset, model: Keras_Model,
        input_mask=None, 
        output_mask=None
    ):
        
        super().__init__(data, model, input_mask=input_mask, output_mask=output_mask)
        
        return
    
    
    def fgsm_step(
        self, 
        x_input, y_input, x_perturbation, 
        epsilon=0.03, targeted=False
    ):
        
        x_v = tf.constant(x_input.astype(np.float32))
        y_in = tf.constant(y_input.astype(np.float32))
        x_perturbation = tf.constant(x_perturbation.astype(np.float32))
        
        with tf.GradientTape() as tape:
            tape.watch(x_perturbation)
            
            prediction = self.model(x_v+x_perturbation)
            
            if not targeted:
                loss_value = -self.adv_loss_outputs(y_in, prediction)
            else:
                loss_value = self.adv_loss_outputs(y_in, prediction)
            self.last_run_loss_values += [tf.reduce_mean(loss_value).numpy()]
            
            # adaptive addition to the actual loss
            # 1e15 is the Langrange constant
            loss_value += 1e15*self.model.cav_index
            
            grads = tape.gradient(loss_value, x_perturbation)
        
        grads_sign = tf.sign(grads)
        x_perturbation_new = x_perturbation - epsilon*grads_sign*self.input_mask
        
        return x_perturbation_new.numpy()
    
    
    def universalize_perturbation(self, x_perturbation):
        '''
        Inputs:
            {x_perturbation}: The adversarial perturbation computed by the attack
            
        Outputs:
            {x_perturbation}: A consistent adversarial perturbation by universalizing them
                the batch of inputs.
        '''
        
        history_length = self.data.data_configuration['history_length']
        
        # tranforming into universal adversarial perturbations for consistency.
        x_perturbation[:,:,:,:, :history_length] = np.mean(
            x_perturbation[:,:,:,:,:history_length], 
            axis=(0,4), keepdims=True
        )
        x_perturbation[:,:,:,:, history_length:] = np.mean(
            x_perturbation[:,:,:,:,history_length:], 
            axis=(0,4), keepdims=True
        )
        
        return x_perturbation
    
    