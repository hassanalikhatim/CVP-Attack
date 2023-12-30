import tensorflow as tf
import numpy as np


from _0_general_ML.model_utils.model import Keras_Model

from _0_general_ML.data_utils.dataset import Dataset



eps = 0


class Attack:
    
    def __init__(
        self, 
        data: Dataset, model: Keras_Model,
        input_mask=None, 
        output_mask=None
    ):
        
        self.data = data
        self.model = model
        
        self.input_mask = np.ones_like(data.x_train()[:1])
        if input_mask:
            self.input_mask = input_mask
        
        self.output_mask = np.ones_like(data.y_train()[:1])
        if output_mask:
            self.output_mask = output_mask
            
        self.last_run_loss_values = []
        
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
            # loss_value += self.model.adaptive*1e15*self.model.cav_index
            
            grads = tape.gradient(loss_value, x_perturbation)
        
        grads_sign = tf.sign(grads)
        x_perturbation_new = x_perturbation - epsilon*grads_sign*self.input_mask
        
        return x_perturbation_new.numpy()
    
    
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
    
    