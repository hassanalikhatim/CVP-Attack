import tensorflow as tf
import numpy as np
import time


from _0_general_ML.data_utils.dataset_cards.taxibj import TaxiBJ

from _1_adversarial_ML.attack import Attack

from _6_cvp_attack_paper._0_cav_detect.keras_model_with_call import Keras_Model_with_call



class CVP_Attack(Attack):
    
    def __init__(
        self, 
        data: TaxiBJ, model: Keras_Model_with_call, 
        input_mask=None, output_mask=None
    ):
        
        self.model = model
        
        super().__init__(
            data, model, 
            input_mask, output_mask
        )
        
        self.name = 'CVP Attack'
        self.history_length = int(data.data_configuration['history_length'])
        
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
                tf.reshape(x_perturbation[:1], self.data.x_test[:1,:1,:,:,:1].shape)
            )
            x_perturbation_outflow.append(
                tf.reshape(x_perturbation_o, self.data.x_test[:1,:1,:,:,:1].shape)
            )
            
        x_perturbation = x_perturbation_inflow * self.history_length
        x_perturbation += x_perturbation_outflow * self.history_length
        
        return tf.concat(x_perturbation, axis=4)
    
    
    def cvp_attack_step(
        self, 
        x_input, y_input, x_perturbation_i,
        epsilon=0.03, targeted=False
    ):
        
        x_v = tf.constant(x_input.astype(np.float32))
        y_in = tf.constant(y_input.astype(np.float32))
        x_perturbation_i = tf.constant(x_perturbation_i.astype(np.float32))
        
        with tf.GradientTape() as tape:
            tape.watch(x_perturbation_i)
            
            prediction = self.model(
                x_v + self.compute_inflow_outflow(x_perturbation_i)
            )
            
            if not targeted:
                loss_value = -self.adv_loss_outputs(y_in, prediction)
            else:
                loss_value = self.adv_loss_outputs(y_in, prediction)
            
            self.last_run_loss_values += [tf.reduce_mean(loss_value).numpy()]
            
            if self.model.defended:
                loss_value += 1e15*self.model.cav_index
            
            grads = tape.gradient(loss_value, x_perturbation_i)
        
        return tf.sign(grads).numpy()
    
    
    def attack(
        self, x_input, y_input, 
        epsilon=0.1,
        epsilon_per_iteration=0.03, 
        iterations=1000,
        targeted=False, 
        **kwargs
    ):
        '''
        Inputs:
            {x_input}: the crowd-flow state input of shape (None, 1, 32, 32, 2*history_length)
            {y_input}: the ground-truth crowd-flow state of shape (None, 32, 32, 2)
            {epsilon}: the maximum allowed perturbation to a pixel
        '''
        
        self.epsilon = epsilon
        
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
            
            # L_inf norm
            x_perturbation[:,0] = np.clip(x_perturbation[:,0], -epsilon, epsilon)
            corrected_perturbations = np.clip(
                x_input + np.expand_dims(x_perturbation[:,:1], axis=-1), 0, 1
            ) - x_input
            
            # We only want to keep the inflow perturbations as the outflow state
            # will be generated later through the {self.compute_inflow_outflow} function.
            corrected_perturbations = corrected_perturbations[:,:,:,:,:int(history)]
            
            # We want all the perturbations to be equal. 
            # Averaging is not a good idea because different {x_inputs} 
            # will require different clipping values.
            corrected_perturbations_sign = np.sign(x_perturbation[:,0])
            corrected_perturbations_min = np.min(
                np.abs(corrected_perturbations), 
                axis=(0, -1)
            )
            x_perturbation[:,0] = corrected_perturbations_min*corrected_perturbations_sign
        
        x_perturbation_real = self.compute_inflow_outflow(x_perturbation)
        
        return np.clip(x_input+x_perturbation_real, 0, 1)
    
    