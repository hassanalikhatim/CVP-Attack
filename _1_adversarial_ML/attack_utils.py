import tensorflow as tf
import numpy as np
import time


from .fgsm import FGSM_Attack
from .pgd import PGD_Attack
from .ifgsm import i_FGSM_Attack

eps = 0

class All_Attacks:
    
    def __init__(
        self, 
        data, model,
        input_mask=None, output_mask=None, 
        epsilon=0.1
    ):
        
        self.attack_dictionary = {
            'fgsm': FGSM_Attack(data, model, input_mask=input_mask, output_mask=output_mask),
            'ifgsm': i_FGSM_Attack(data, model, input_mask=input_mask, output_mask=output_mask),
            'pgd': PGD_Attack(data, model, input_mask=input_mask, output_mask=output_mask),
            'upa': self.universal_physical_attack
        }
        self.model = model
        self.data = data
        
        if input_mask is None:
            self.input_mask = np.ones_like(data.x_train()[:1])
        else:
            self.input_mask = input_mask
        
        if output_mask is None:
            self.output_mask = np.ones_like(data.y_train()[:1])
        else:
            self.output_mask = output_mask
        
        self.last_run_loss_values = []
    
    def adv_loss_outputs(self, y_true, y_pred):
        if self.data.num_classes is None:
            loss = tf.reduce_sum(tf.square(y_true - y_pred)*self.output_mask, axis=0)
        else:
            loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            loss = -tf.reduce_sum(loss*self.output_mask, axis=0)
        return loss
    
    def adv_loss_inputs(self, x_delta, loss_type=2):        
        loss = {
            2: tf.reduce_mean(tf.square(x_delta), axis=[1,2,3]),
            1: tf.reduce_mean(tf.abs(x_delta), axis=[1,2,3])
        }
        return loss[loss_type]
    
    def fgsm_step(self, x_input, y_input, x_perturbation, 
                  epsilon=0.03, targeted=False):
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
            loss_value += self.model.adaptive*1e15*self.model.cav_index
            
            grads = tape.gradient(loss_value, x_perturbation)
        
        grads_sign = tf.sign(grads)
        x_perturbation_new = x_perturbation - epsilon*grads_sign*self.input_mask
        return x_perturbation_new.numpy()
    
    def create_upa_adjacency_matrix(self):
        from utils.general_utils import neighbour_image_pixels
        (h, w) = self.data.x[0,0,:,:,0].shape
        self.upa_adjacency_matrix = np.zeros((h*w, h*w)).astype('float32')
        for k in np.arange(h*w):
            neighbours = neighbour_image_pixels(k, h, w, num_neighbour=2)
            self.upa_adjacency_matrix[k, neighbours] = 1.
        one_matrix = np.eye(self.upa_adjacency_matrix.shape[0])
        self.upa_adjacency_matrix = self.upa_adjacency_matrix * (1-one_matrix)
        
    def _compute_inflow_outflow(self, x_perturbations):
        x_perturbation_inflow = []
        x_perturbation_outflow = []
        for x_perturbation in x_perturbations:
            weight_matrix = tf.keras.activations.sigmoid(x_perturbation[1:])
            x_perturbation_o = tf.math.multiply(weight_matrix, self.upa_adjacency_matrix)/(8+7)
            x_perturbation_o = tf.matmul(x_perturbation[:1], x_perturbation_o)
            x_perturbation_inflow.append(tf.reshape(x_perturbation[:1], self.data.x_test()[:1,:1,:,:,:1].shape))
            x_perturbation_outflow.append(tf.reshape(x_perturbation_o, self.data.x_test()[:1,:1,:,:,:1].shape))
        return x_perturbation_inflow, x_perturbation_outflow
    
    def universal_adversarial_perturbation(self, x_input, y_input,
                                           epsilon=0.1, num_devices=15000,
                                           epsilon_per_iteration=0.03, 
                                           iterations=1000,
                                           targeted=False, **kwargs):
        self.last_run_loss_values = []
        previous_print_time = 0
        x_perturbation = np.zeros_like(x_input[:1]).astype(np.float32)
        for iteration in range(iterations):
            if time.time() - previous_print_time > 1:
                previous_print_time = time.time()
                print("\rAttacking: ", iteration, "/", iterations, "\t", end="")
            x_perturbation = self.fgsm_step(x_input, y_input, x_perturbation, 
                                            epsilon=epsilon_per_iteration,
                                            targeted=targeted)
            x_perturbation = np.clip(x_perturbation, -epsilon, epsilon)
        return np.clip(x_input+x_perturbation, eps, 1-eps)
    
    def evaluate_untargeted(self, epsilon=0.1, N=100, attack_name='fgsm'):
        x_adversarials = self.attack_dictionary[attack_name](self.data.x_test()[-N:], 
                                                             self.data.y_test()[-N:],
                                                             epsilon=epsilon)
        x_adversarials = np.clip(x_adversarials, eps, 1-eps)
        
        if self.data.num_classes is not None:
            y_clean = np.argmax(self.model.predict(self.data.x_test()[-N:]), axis=-1)
            y_adversarials = np.argmax(self.model.predict(x_adversarials), axis=-1)
            y_test = np.argmax(self.data.y_test()[-N:], axis=-1)
        else:
            y_clean, _ = self.data.get_classified_data(self.model.predict(self.data.x_test()[-N:]), num_classes=10)
            y_clean = np.argmax(y_clean, axis=-1)
            y_adversarials, _ = self.data.get_classified_data(self.model.predict(x_adversarials), num_classes=10)
            y_adversarials = np.argmax(y_adversarials, axis=-1)
            y_test, _ = self.data.get_classified_data(self.data.y_test()[-N:], num_classes=10)
            y_test = np.argmax(y_test, axis=-1)
        
        clean_acc = np.mean(y_test==y_clean)
        adv_acc = np.mean(y_test==y_adversarials)
        clean_loss = np.mean((y_clean/10-y_test/10)**2)
        adv_loss = np.mean((y_adversarials/10-y_test/10)**2)
            
        print("Clean Metrics: ", clean_acc, clean_loss)
        print("Adv. Metrics: ", adv_acc, adv_loss)
        return clean_acc, clean_loss, adv_acc, adv_loss