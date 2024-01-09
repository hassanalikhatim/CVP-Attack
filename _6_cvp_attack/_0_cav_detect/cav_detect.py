import numpy as np
import tensorflow as tf


from _0_general_ML.data_utils.dataset_cards.taxibj import TaxiBJ



class CaV_Detect:
    
    def __init__(
        self, 
        data: TaxiBJ, model: tf.keras.Model,
        y_target=None, 
        n=2
    ):
        
        self.data = data
        self.model = model
        
        self.n = n
        
        self.defended = True
        
        return
    
    
    def consistency_check_on_batch(self, x_input):
        
        history_length = self.data.data_configuration['history_length']
        
        indices_current = np.arange(history_length-1)
        indices_current = np.append(
            indices_current, 
            np.arange(history_length, history_length*2-1),
            axis=0
        )
        indices_previous = indices_current + 1
        
        difference = tf.stack([x_input[1:, 0, :, :, i] for i in indices_current], axis=-1)
        difference -= tf.stack([x_input[:-1, 0, :, :, i] for i in indices_previous], axis=-1)
        difference = tf.concat([tf.zeros_like(difference[:1]), difference], axis=0)

        # difference = tf.math.tanh(
        #     tf.reduce_sum(tf.abs(difference), axis=(1,2,3), keepdims=True)
        # )
        difference = tf.reduce_sum(tf.abs(difference), axis=(1,2,3), keepdims=True)
        
        return difference
    
    
    def convolve_3d(self, test_input, filter_):
        
        history_size = int(test_input.shape[-1]/2)
        
        sum_correlations = []
        for i in range(history_size):
            test_input_slice = tf.transpose(
                tf.stack(
                    [test_input[:,0,:,:,k] for k in [i,i+history_size]]
                ),
                (1,2,3,0)
            )
            correlation_1 = tf.nn.conv2d(test_input_slice, filter_, 1, 'SAME')
            sum_correlations.append(correlation_1)
        
        sum_correlations = tf.concat(sum_correlations, axis=3)
        
        return sum_correlations
    
    
    def validity_check_on_batch(self, x_input):
        
        filter_size = ( 2 * self.n ) + 1
        filter_in = np.ones((filter_size, filter_size, 1, 1)).astype(np.float32)
        filter_in[int(filter_size/2), int(filter_size/2)] = 0
        filter_out = filter_in - 1
        filter_0 = tf.constant(np.append(filter_in, filter_out, axis=2).astype(np.float32))
        filter_1 = tf.constant(np.append(filter_out, filter_in, axis=2).astype(np.float32))

        sum_correlation_0 = tf.nn.relu(-self.convolve_3d(x_input, filter_0))
        sum_correlation_1 = tf.nn.relu(-self.convolve_3d(x_input, filter_1))
        
        inflow_validity = tf.reduce_sum(sum_correlation_0, axis=(1,2,3), keepdims=True)
        outflow_validity = tf.reduce_sum(sum_correlation_1, axis=(1,2,3), keepdims=True)
        # inflow_validity = tf.math.tanh(
        #     tf.reduce_sum(sum_correlation_0, axis=(1,2,3), keepdims=True)
        # )
        # outflow_validity = tf.math.tanh(
        #     tf.reduce_sum(sum_correlation_1, axis=(1,2,3), keepdims=True)
        # )
        
        return inflow_validity + outflow_validity
    
    
    def call(self, x_input):
        
        self.cav_index = self.validity_check_on_batch(1000*x_input)
        self.cav_index += self.consistency_check_on_batch(1000*x_input)

        return self.model(x_input)
    
    
    def validity_detector(self, x_input):
        return (self.validity_check_on_batch(1000*x_input) >= 1)
    def consistency_detector(self, x_input):
        return (self.consistency_check_on_batch(1000*x_input) >=1 )
    def detect(self, x_input):
        gamma_v = self.validity_check_on_batch(1000*x_input)
        gamma_c = self.consistency_check_on_batch(1000*x_input)
        return (gamma_c > 1), (gamma_v > 1), ((gamma_c > 1) | (gamma_v > 1))

    def predict(self, x_input):
        return self(x_input.astype(np.float32)).numpy()
    def __call__(self, x_input):
        return self.call(x_input)
    def evaluate(self, x_input, y_input, verbose=True, **kwargs):
        return self.model.evaluate(x_input, y_input, verbose=verbose)
