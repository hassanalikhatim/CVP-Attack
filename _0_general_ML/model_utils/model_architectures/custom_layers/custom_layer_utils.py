import tensorflow as tf
from tensorflow.keras import activations
import numpy as np

class Image(tf.keras.layers.Layer):
    def __init__(self, out_sample):
        self.out_shape = out_sample.shape[1:]
        super(Image, self).__init__()

    def build(self, input_shape):
        self.image = self.add_weight(shape=self.out_shape, initializer="random_normal", trainable=True)
        self.values = self.add_weight(shape=input_shape, initializer='random_normal', trainable=True)
      
    def call(self, inputs):
        multiplier = tf.reduce_sum(tf.math.multiply(self.values, inputs))
        return tf.keras.backend.clip(self.image*multiplier, 0, 1)
        
        
class Custom_Dense(tf.keras.layers.Layer):
    def __init__(self, out_neurons, activation='linear', kernel_regularizer=None):
        self.out_neurons = out_neurons
        self.activation = tf.keras.activations.get(activation)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        super(Custom_Dense, self).__init__()

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=list(input_shape[1:])+[self.out_neurons], initializer="random_normal", 
                                     regularizer=self.kernel_regularizer, trainable=True)
      
    def call(self, inputs):
        return self.activation(tf.keras.backend.dot(inputs, self.kernel) - 0.5)
        

class STResNet_Unit(tf.keras.layers.Layer):
    def __init__(self, filters=64, kernel_size=(7,7), kernel_regularizer=None, name=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        super(STResNet_Unit, self).__init__(name=name)
    
    def build(self, input_shape):
        kernel_shape = [self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.filters]
        self.kernel_1 = self.add_weight(shape=kernel_shape, initializer="random_normal", 
                                        regularizer=self.kernel_regularizer, trainable=True, name='kernel_1')
        self.bias_1 = self.add_weight(shape=(self.filters,), initializer="random_normal", 
                                      regularizer=self.kernel_regularizer, trainable=True, name='bias_1')
        kernel_shape[2] = self.filters
        self.kernel_2 = self.add_weight(shape=kernel_shape, initializer="random_normal", 
                                        regularizer=self.kernel_regularizer, trainable=True, name='kernel_2')
        self.bias_2 = self.add_weight(shape=(self.filters,), initializer="random_normal", 
                                      regularizer=self.kernel_regularizer, trainable=True, name='bias_2')
        # self.multiplier = self.add_weight(shape=[1]+list(input_shape[1:]), initializer="random_normal",
        #                                   trainable=True, name="multiplier")
    
    def call(self, inputs):
        out = tf.nn.conv2d(inputs, self.kernel_1, strides=1, padding='SAME') + self.bias_1
        out = tf.keras.activations.relu(out)
        out = tf.nn.conv2d(out, self.kernel_2, strides=1, padding='SAME') + self.bias_2
        out = tf.keras.activations.relu(out)
        out = out + inputs
        # out = out + self.multiplier*inputs
        return out


class Fusion(tf.keras.layers.Layer):
    def __init__(self, kernel_regularizer=None):
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        super(Fusion, self).__init__()
    
    def build(self, input_shape):
        print("Fusion input shape: ", input_shape)
        self.w_1 = self.add_weight(shape=[1]+list(input_shape[0][1:]), initializer="random_normal", 
                                   regularizer=self.kernel_regularizer, trainable=True, name="w_1")
        self.w_2 = self.add_weight(shape=[1]+list(input_shape[1][1:]), initializer="random_normal", 
                                   regularizer=self.kernel_regularizer, trainable=True, name="w_2")
        self.w_3 = self.add_weight(shape=[1]+list(input_shape[2][1:]), initializer="random_normal", 
                                   regularizer=self.kernel_regularizer, trainable=True, name="w_3")
        self.bias = self.add_weight(shape=[1]+list(input_shape[2][1:]), initializer="random_normal", 
                                    trainable=True, name="bias")
    
    def call(self, inputs):
        closeness, period, trend = inputs
        out = tf.math.multiply(closeness, self.w_1) + tf.math.multiply(period, self.w_2) + tf.math.multiply(trend, self.w_3) + self.bias
        out = tf.keras.activations.tanh(out)
        return out
    

class Gated_Graph_Layer(tf.keras.layers.Layer):
    def __init__(self, hidden_dimension=10, message_dimension=5, time_steps=1,
                 adjacency_matrix=None, kernel_regularizer=None, message_types=1):
        self.hidden_dimension = hidden_dimension
        self.message_dimension = message_dimension
        self.time_steps = time_steps
        self.adj = adjacency_matrix
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.message_types = message_types
        super(Gated_Graph_Layer, self).__init__()
        
    def build(self, input_shape):
        print("Input shape: ", input_shape)
        self.num_nodes, input_dimensions = input_shape[1:]
        if self.adj is None:
            self.adjacency_matrix = self.add_weight(shape=(self.num_nodes, self.num_nodes),
                                                    initializer='zeros',
                                                    regularizer=self.kernel_regularizer,
                                                    trainable=True, name="adjency_matrix")
        else:
            self.adjacency_matrix = tf.constant(self.adj)
        self.message_weight = self.add_weight(shape=(2+self.hidden_dimension, self.message_dimension),
                                              initializer='random_normal',
                                              regularizer=self.kernel_regularizer,
                                              trainable=True, name="message_weight")
        self.message_bias = self.add_weight(shape=(1, self.message_dimension, 1),
                                            initializer='zeros',
                                            trainable=True, name="message_bias")
        self.gate_1_message = self.add_weight(shape=(self.message_dimension, self.hidden_dimension),
                                              initializer='random_normal',
                                              trainable=True, name="gate_1_message")
        self.gate_1_hidden = self.add_weight(shape=(self.hidden_dimension, self.hidden_dimension),
                                             initializer='random_normal',
                                             trainable=True, name="gate_1_hidden")
        self.gate_2_message = self.add_weight(shape=(self.message_dimension, self.hidden_dimension),
                                              initializer='random_normal',
                                              trainable=True, name="gate_2_message")
        self.gate_2_hidden = self.add_weight(shape=(self.hidden_dimension, self.hidden_dimension),
                                             initializer='random_normal',
                                             trainable=True, name="gate_2_hidden")
        self.output_message = self.add_weight(shape=(self.message_dimension, self.hidden_dimension),
                                              initializer='random_normal',
                                              trainable=True, name="output_message")
        self.output_hidden = self.add_weight(shape=(self.hidden_dimension, self.hidden_dimension),
                                             initializer='random_normal',
                                             trainable=True, name="output_hidden")
    
    def compute_message_out(self, input_slice, hidden_state):
        input_hidden = tf.concat([input_slice, hidden_state], axis=-1)
        message_out = tf.matmul(input_hidden, self.message_weight)
        message_out = tf.transpose(message_out, (0, 2, 1))
        message_out = tf.matmul(message_out, self.adjacency_matrix) + self.message_bias
        message_out = tf.transpose(message_out, (0, 2, 1))
        message_out = tf.keras.activations.sigmoid(message_out)
        return message_out
    
    def compute_gate_1(self, message_out, hidden_state):
        gate_1 = tf.matmul(message_out, self.gate_1_message)
        gate_1 += tf.matmul(hidden_state, self.gate_1_hidden)
        return tf.keras.activations.sigmoid(gate_1)
    
    def compute_gate_2(self, message_out, hidden_state):
        gate_2 = tf.matmul(message_out, self.gate_2_message)
        gate_2 += tf.matmul(hidden_state, self.gate_2_hidden)
        return tf.keras.activations.sigmoid(gate_2)
    
    def initial_next_state(self, message_out, gated_hidden_state):
        next_state = tf.matmul(message_out, self.output_message)
        next_state += tf.matmul(gated_hidden_state, self.output_hidden)
        return tf.keras.activations.tanh(next_state)
    
    def call(self, inputs):
        timer = int(inputs.shape[-1]/2)
        hidden_state = tf.zeros_like(inputs[:,:,:1]) 
        # hidden_state.shape => [None, 1024, 1]
        hidden_state += tf.zeros(shape=[1]*len(inputs.shape[:-1])+[self.hidden_dimension]) 
        # hidden_state.shape => [None, 1024, hidden_dimension]
        for k in range(timer):
            input_slice = tf.concat([inputs[:,:,k:k+1], inputs[:,:,timer+k:timer+k+1]], axis=-1)
            message_out = self.compute_message_out(input_slice, hidden_state)
            gate_1 = self.compute_gate_1(message_out, hidden_state)
            gate_2 = self.compute_gate_2(message_out, hidden_state)
            next_state = self.initial_next_state(message_out, tf.math.multiply(gate_1, hidden_state))
            hidden_state = tf.math.multiply(1-gate_2, hidden_state)
            hidden_state += tf.math.multiply(gate_2, next_state)
        return hidden_state 

class NiN_Conv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(7,7), input_shape=None, nin_depths=[], kernel_regularizer=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.nin_depths = nin_depths + [1]
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        super(NiN_Conv2D, self).__init__()
        
    def build(self, input_shape):
        print("Input shape: ", input_shape)
        self.paddings = tf.constant([[0, 0], [int((self.kernel_size[0])/2), int((self.kernel_size[0])/2)],
                                     [int((self.kernel_size[1])/2), int((self.kernel_size[1])/2)], [0, 0]])
        self.nin_dense = [self.add_weight(shape=(1, self.kernel_size[0]*self.kernel_size[1]*input_shape[-1], self.nin_depths[0], self.filters), 
                                          initializer=tf.keras.initializers.Ones(), regularizer=self.kernel_regularizer, trainable=True)]
        self.nin_bias = [self.add_weight(shape=(1, self.nin_depths[0], 1, self.filters), 
                                         initializer=tf.keras.initializers.Zeros(), trainable=True)]
        for n in range(1, len(self.nin_depths)):
            self.nin_dense.append(self.add_weight(shape=(1, self.nin_depths[n-1], self.nin_depths[n], self.filters), 
                                                  initializer=tf.keras.initializers.Ones(), regularizer=self.kernel_regularizer, trainable=True))
            self.nin_bias.append(self.add_weight(shape=(1, self.nin_depths[n], 1, self.filters), 
                                                 initializer=tf.keras.initializers.Zeros(), trainable=True))
        
    def call(self, inputs):
        inputs = tf.pad(inputs, self.paddings, 'CONSTANT')
        # print("Input shape: ", inputs[:, :self.kernel_size[0], :self.kernel_size[1], :].shape)
        conv_ = []
        for row in range(inputs.shape[1]-self.kernel_size[0]+1):
            conv_row = []
            for col in range(inputs.shape[2]-self.kernel_size[1]+1):
                product = inputs[:, row:row+self.kernel_size[0], col:col+self.kernel_size[1], :]
                product_shape = (-1, self.kernel_size[0]*self.kernel_size[1]*inputs.shape[-1], 1, 1)
                product = tf.reshape(product, shape=product_shape)
                for d, dense in enumerate(self.nin_dense):
                    product = tf.math.multiply(product, dense)
                    product = tf.expand_dims(tf.reduce_sum(product, axis=-3), axis=2) + self.nin_bias[d]
                    product = tf.keras.activations.relu(product)
                conv_row.append(product[:,0,0,:])
            conv_.append(conv_row)
        conv_ = tf.stack(conv_)
        conv_ = tf.transpose(conv_, (2,0,1,3))
        return conv_
    
    def _call(self, inputs):
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], inputs.shape[-1], self.filters)
        kernel = np.ones(kernel_shape)
        conv_ = tf.nn.conv2d(inputs, kernel, strides=1, padding='VALID')
        return tf.keras.activations.relu(conv_)
    