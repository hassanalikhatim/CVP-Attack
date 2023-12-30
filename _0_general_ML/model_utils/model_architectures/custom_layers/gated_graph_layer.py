import tensorflow as tf



class Gated_Graph_Layer(tf.keras.layers.Layer):
    
    def __init__(
        self, 
        hidden_dimension=10, message_dimension=5, 
        time_steps=1,
        adjacency_matrix=None, 
        kernel_regularizer=None, 
        message_types=1
    ):
        
        self.hidden_dimension = hidden_dimension
        self.message_dimension = message_dimension
        
        self.time_steps = time_steps
        self.adj = adjacency_matrix
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.message_types = message_types
        
        # super(Gated_Graph_Layer, self).__init__()
        super().__init__()
        
        return
    
        
    def build(self, input_shape):
        
        print("Input shape: ", input_shape)
        
        self.num_nodes, input_dimensions = input_shape[1:]
        
        if self.adj is None:
            self.adjacency_matrix = self.add_weight(
                shape=(self.num_nodes, self.num_nodes),
                initializer='zeros',
                regularizer=self.kernel_regularizer,
                trainable=True, name="adjency_matrix"
            )
        else:
            self.adjacency_matrix = tf.constant(self.adj)
        
        self.message_weight = self.add_weight(
            shape=(2+self.hidden_dimension, self.message_dimension),
            initializer='random_normal',
            regularizer=self.kernel_regularizer,
            trainable=True, name="message_weight"
        )
        self.message_bias = self.add_weight(
            shape=(1, self.message_dimension, 1),
            initializer='zeros',
            trainable=True, name="message_bias"
        )
        self.gate_1_message = self.add_weight(
            shape=(self.message_dimension, self.hidden_dimension),
            initializer='random_normal',
            trainable=True, name="gate_1_message"
        )
        self.gate_1_hidden = self.add_weight(
            shape=(self.hidden_dimension, self.hidden_dimension),
            initializer='random_normal',
            trainable=True, name="gate_1_hidden"
        )
        self.gate_2_message = self.add_weight(
            shape=(self.message_dimension, self.hidden_dimension),
            initializer='random_normal',
            trainable=True, name="gate_2_message"
        )
        self.gate_2_hidden = self.add_weight(
            shape=(self.hidden_dimension, self.hidden_dimension),
            initializer='random_normal',
            trainable=True, name="gate_2_hidden"
        )
        self.output_message = self.add_weight(
            shape=(self.message_dimension, self.hidden_dimension),
            initializer='random_normal',
            trainable=True, name="output_message"
        )
        self.output_hidden = self.add_weight(
            shape=(self.hidden_dimension, self.hidden_dimension),
            initializer='random_normal',
            trainable=True, name="output_hidden"
        )
        
        return
    
    
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
        
        # hidden_state.shape => [None, 1024, 1]
        hidden_state = tf.zeros_like(inputs[:,:,:1])
        
        # hidden_state.shape => [None, 1024, hidden_dimension]
        hidden_state += tf.zeros(shape=[1]*len(inputs.shape[:-1])+[self.hidden_dimension]) 
        for k in range(timer):
            input_slice = tf.concat(
                [inputs[:,:,k:k+1], inputs[:,:,timer+k:timer+k+1]], 
                axis=-1
            )
            
            message_out = self.compute_message_out(input_slice, hidden_state)
            gate_1 = self.compute_gate_1(message_out, hidden_state)
            gate_2 = self.compute_gate_2(message_out, hidden_state)
            next_state = self.initial_next_state(message_out, tf.math.multiply(gate_1, hidden_state))
            
            hidden_state = tf.math.multiply(1-gate_2, hidden_state)
            hidden_state += tf.math.multiply(gate_2, next_state)
        
        return hidden_state 