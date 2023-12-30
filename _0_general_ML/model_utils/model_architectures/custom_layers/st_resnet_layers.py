import tensorflow as tf



class STResNet_Unit(tf.keras.layers.Layer):
    
    def __init__(
        self, 
        filters=64, kernel_size=(7,7), 
        kernel_regularizer=None, 
        name=None
    ):
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        
        # super(STResNet_Unit, self).__init__(name=name)
        super().__init__()
        
        return
    
    
    def build(self, input_shape):
        
        kernel_shape = [self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.filters]
        
        self.kernel_1 = self.add_weight(
            shape=kernel_shape, initializer="random_normal", 
            regularizer=self.kernel_regularizer, 
            trainable=True, name='kernel_1'
        )
        self.bias_1 = self.add_weight(
            shape=(self.filters,), initializer="random_normal", 
            regularizer=self.kernel_regularizer, 
            trainable=True, name='bias_1'
        )
        
        kernel_shape[2] = self.filters
        self.kernel_2 = self.add_weight(
            shape=kernel_shape, initializer="random_normal", 
            regularizer=self.kernel_regularizer, 
            trainable=True, name='kernel_2'
        )
        self.bias_2 = self.add_weight(
            shape=(self.filters,), initializer="random_normal", 
            regularizer=self.kernel_regularizer, 
            trainable=True, name='bias_2'
        )
        
        return
    
    
    def call(self, inputs):
        out = tf.nn.conv2d(inputs, self.kernel_1, strides=1, padding='SAME') + self.bias_1
        out = tf.keras.activations.relu(out)
        out = tf.nn.conv2d(out, self.kernel_2, strides=1, padding='SAME') + self.bias_2
        out = tf.keras.activations.relu(out)
        out = out + inputs
        return out



class Fusion(tf.keras.layers.Layer):
    
    def __init__(
        self, kernel_regularizer=None
    ):
        
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        
        # super(Fusion, self).__init__()
        super().__init__()
        
        return
    
    
    def build(self, input_shape):
        
        print("Fusion input shape: ", input_shape)
        
        self.w_1 = self.add_weight(
            shape=[1]+list(input_shape[0][1:]), 
            initializer="random_normal", 
            regularizer=self.kernel_regularizer, 
            trainable=True, name="w_1"
        )
        self.w_2 = self.add_weight(
            shape=[1]+list(input_shape[1][1:]), 
            initializer="random_normal", 
            regularizer=self.kernel_regularizer, 
            trainable=True, name="w_2"
        )
        self.w_3 = self.add_weight(
            shape=[1]+list(input_shape[2][1:]), 
            initializer="random_normal", 
            regularizer=self.kernel_regularizer,
            trainable=True, name="w_3"
        )
        self.bias = self.add_weight(
            shape=[1]+list(input_shape[2][1:]), 
            initializer="random_normal", 
            trainable=True, name="bias"
        )
        
        return
    
    
    def call(self, inputs):
        closeness, period, trend = inputs
        
        out = tf.math.multiply(closeness, self.w_1) 
        out += tf.math.multiply(period, self.w_2) 
        out += tf.math.multiply(trend, self.w_3) 
        out += self.bias
        
        out = tf.keras.activations.tanh(out)
        
        return out