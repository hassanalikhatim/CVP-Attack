import tensorflow as tf
import numpy as np
import time


from _0_general_ML.data_utils.dataset_cards.taxibj import TaxiBJ
from _0_general_ML.model_utils.model import Keras_Model

from _1_adversarial_attacks.attack import Attack



class CVP_Attack(Attack):
    
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
    
    
    def compute_inflow_outflow(self, x_perturbations):
        
        x_perturbation_inflow = []
        x_perturbation_outflow = []
        for x_perturbation in x_perturbations:
            weight_matrix = tf.keras.activations.sigmoid(x_perturbation[1:])
            weight_matrix = weight_matrix/np.sum(weight_matrix, axis=0)
            x_p_os = tf.math.multiply(weight_matrix, x_perturbation[:1])
            
            n = 2
            paddings = tf.constant([[0,0], [n, n], [n, n]])
            x_p_os = tf.pad(x_p_os, paddings, "CONSTANT")
            tensor_list = []
            for i in range(-n, n+1):
                for j in range(-n, n+1):
                    k = (2*n+1)*(i+n)+j+n
                    if i>0 and j>0:
                        k = k-1
                    if i!=0 or j!=0:
                        tensor_list.append(x_p_os[k:k+1, n-i:n-i+32, n-j:n-j+32])
            x_perturbation_o = tf.concat(tensor_list, axis=0)
            x_perturbation_o = tf.reduce_sum(x_perturbation_o, axis=0)
            
            x_perturbation_inflow.append(
                tf.reshape(x_perturbation[:1], self.data.x_test()[:1,:1,:,:,:1].shape)
            )
            x_perturbation_outflow.append(
                tf.reshape(x_perturbation_o, self.data.x_test()[:1,:1,:,:,:1].shape)
            )
        
        return x_perturbation_inflow, x_perturbation_outflow
    
    
    def upa_step(
        self, 
        x_input, y_input, x_perturbation_i,
        epsilon=0.03, targeted=False
    ):
        
        x_v = tf.constant(x_input.astype(np.float32))
        y_in = tf.constant(y_input.astype(np.float32))
        x_perturbation_i = tf.constant(x_perturbation_i.astype(np.float32))
        
        with tf.GradientTape() as tape:
            tape.watch(x_perturbation_i)
            
            x_perturbation_inflow, x_perturbation_outflow = self.compute_inflow_outflow(x_perturbation_i)
            
            if x_perturbation_i.shape[0] == 1:
                x_perturbation = x_perturbation_inflow*int(x_v.shape[-1]/2) + x_perturbation_outflow*int(x_v.shape[-1]/2)
                x_perturbation = tf.concat(x_perturbation, axis=4)
            else:
                x_perturbation = tf.concat(x_perturbation_inflow + x_perturbation_outflow, axis=4)
            prediction = self.model(x_v+x_perturbation)
            
            if not targeted:
                loss_value = -self.adv_loss_outputs(y_in, prediction)
            else:
                loss_value = self.adv_loss_outputs(y_in, prediction)
            self.last_run_loss_values += [tf.reduce_mean(loss_value).numpy()]
            loss_value += self.model.adaptive*1e15*self.model.cav_index
            
            grads = tape.gradient(loss_value, x_perturbation_i)
        
        return tf.sign(grads).numpy()
    
    
    def attack(
        self, x_input, y_input, 
        epsilon=0.1, norm='li',
        epsilon_per_iteration=0.03, 
        iterations=1000,
        targeted=False, 
        **kwargs
    ):
        
        history = self.data.data_configuration['history']
        self.last_run_loss_values = []
        previous_print_time = 0
        epsilon_per_iteration = epsilon/(iterations/4)
        
        # x_perturbation = -5*np.ones([history, 9]+list(x_input.shape[2:4])).astype(np.float32)
        x_perturbation = -5*np.ones([1, 25]+list(x_input.shape[2:4])).astype(np.float32)
        x_perturbation[:,0] = 0.
        for iteration in range(iterations):
            # Just for the progress bar to run
            if time.time() - previous_print_time > 1:
                previous_print_time = time.time()
                print("\rAttacking: ", iteration, "/", iterations, "\t", end="")
            
            p_perturbation = self.upa_step(
                x_input, y_input, x_perturbation,
                epsilon=epsilon_per_iteration,
                targeted=targeted
            )
            x_perturbation[:,:1] -= epsilon_per_iteration*p_perturbation[:,:1]
            x_perturbation[:,1:] -= (10/iterations)*p_perturbation[:,1:]
            
            if norm=='li':
                x_perturbation[:,0] = np.clip(x_perturbation[:,0], -epsilon, epsilon)
                corrected_perturbations = np.clip(x_input+np.expand_dims(x_perturbation[:,:1], axis=-1), 0, 1) - x_input
                corrected_perturbations = corrected_perturbations[:,:,:,:,:int(history/2)]
                corrected_perturbations_sign = np.sign(x_perturbation[:,0])
                corrected_perturbations_min = np.min(np.abs(corrected_perturbations), axis=(0, -1))
                x_perturbation[:,0] = corrected_perturbations_min*corrected_perturbations_sign
                # print(np.min(x_perturbation), np.min(x_perturbation[:,:1]))
            elif norm=='l1':
                x_perturbation[:,0] = np.clip(x_perturbation[:,0], 0, 1)
                current_l1_norm = np.expand_dims(
                    np.mean(np.abs(x_perturbation[:,0]), axis=(1,2)), 
                    axis=(0,2,3)
                )
                if np.max(current_l1_norm) > epsilon:
                    indices = np.where( current_l1_norm > epsilon )
                    x_perturbation[indices,0] *= epsilon / current_l1_norm[:,indices]
        
        x_p_in, x_p_out = self.compute_inflow_outflow(x_perturbation)
        if x_perturbation.shape[0] == 1:
            x_perturbation_real = x_p_in*int(x_input.shape[-1]/2) + x_p_out*int(x_input.shape[-1]/2)
            x_perturbation_real = tf.concat(x_perturbation_real, axis=4)
        else:
            x_perturbation_real = tf.concat(x_p_in + x_p_out, axis=4)
        
        return np.clip(x_input+x_perturbation_real, 0, 1)
    
    