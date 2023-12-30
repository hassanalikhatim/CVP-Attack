import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Input, UpSampling2D
from tensorflow.keras.layers import Flatten, Multiply, Activation, Reshape
from _0_general_ML.model_utils.model_architectures.custom_layers.custom_layer_utils import Image, Custom_Dense, STResNet_Unit, Fusion
from _0_general_ML.model_utils.model_architectures.custom_layers.gated_graph_layer import Gated_Graph_Layer
from _0_general_ML.model_utils.model_architectures.custom_layers.custom_layer_utils import NiN_Conv2D
import numpy as np


class Custom_Loss:
    def __init__(self, r=0.5, out_sample_shape=(1,30,30,10), class_weights=None):
        self.r = r
        self.class_weights = np.ones(out_sample_shape).astype('float32')
        for w, weight in enumerate(class_weights):
            self.class_weights[:,:,:,w] = class_weights[w]
        print("update 2 :", self.class_weights.shape)
    
    def road_state_loss(self, y_true, y_pred):
        state_difference = tf.reduce_mean(tf.square(y_true[0] - y_pred[0]), axis=-1)
        road_difference = tf.reduce_mean(tf.square(y_true[1] - y_pred[1]), axis=-1)
        # road_state_difference = tf.reduce_mean(tf.square(y_pred[0] - y_pred[1]), axis=-1)
        return (1-self.r)*state_difference + self.r*road_difference
        
    def weighted_categorical_crossentropy(self, y_true, y_pred):
        target = tf.math.multiply(self.class_weights, y_true)
        cce_1 = tf.math.multiply(y_true, tf.math.log(y_pred))
        cce_2 = tf.math.multiply(1-y_true, tf.math.log(1-y_pred))
        loss = -tf.reduce_sum(tf.math.multiply(target, cce_1+cce_2), axis=-1)
        # loss = tf.keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1)
        return tf.reduce_mean(loss, axis=0)


def image_model(in_sample, out_sample, learning_rate=1e-4):
    model = tf.keras.models.Sequential()
    model.add(Image(out_sample))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model


def nin_baseline(data, parameters=[3, 1, 0], regularizer=None, 
                 learning_rate=1e-4):
    in_samples, out_samples = data.x_train()[:1], data.y_train()[:1]
    model = tf.keras.Sequential()
    model.add(Input(shape=in_samples.shape[1:]))
    model.add(Reshape(target_shape=list(in_samples.shape[2:-1])+[-1]))
    for r in range(parameters[0]):
        model.add(NiN_Conv2D(32, kernel_size=(7,7)))
    model.add(Flatten())
    model.add(Dense(out_samples[0].reshape(-1).shape[0], kernel_regularizer=regularizer))
    model.add(Reshape(target_shape=out_samples.shape[1:]))
    
    if data.num_classes is None:
        model.add(Activation('sigmoid'))
        loss = 'mse'
        metrics = None
    else:
        model.add(Activation('softmax'))
        loss = 'categorical_crossentropy'
        metrics = ['acc']
        
    model.compile(loss=loss, metrics=metrics, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    model.summary()
    return model