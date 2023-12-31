import tensorflow as tf
import numpy as np


from _0_general_ML.data_utils.dataset_cards.taxibj import TaxiBJ

from .custom_layers.gated_graph_layer import Gated_Graph_Layer

from utils_.general_utils import prepare_adjacency_matrix



def TGCN_baseline(
    data: TaxiBJ, model_configuration: dict,
    **kwargs
):
    
    default_model_configuration = {
        'hidden_layers': 1,
        'message_dimensions': 10,
        'adjacent_nodes': 3,
        'weight_decay': None,
        'learning_rate': 1e-4
    }
    for key in model_configuration.keys():
        default_model_configuration[key] = model_configuration[key]
    
    in_samples, out_samples = data.x_train[:1], data.y_train[:1]
    
    adjacency_matrix = prepare_adjacency_matrix(
        in_samples[0], default_model_configuration['adjacent_nodes']
    )
    
    
    # model definition starts here
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=in_samples[0].shape))
    model.add(tf.keras.layers.Reshape(target_shape=[-1, in_samples.shape[-1]]))
    
    for r in range(default_model_configuration['hidden_layers']):
        model.add(
            Gated_Graph_Layer(
                message_dimension=default_model_configuration['message_dimensions'], 
                adjacency_matrix=adjacency_matrix, 
                time_steps=data.data_configuration['history']
            )
        )
    model.add(tf.keras.layers.Flatten())
    
    model.add(
        tf.keras.layers.Dense(
            out_samples[0].reshape(-1).shape[0], 
            kernel_regularizer=default_model_configuration['weight_decay']
        )
    )
    model.add(tf.keras.layers.Reshape(target_shape=out_samples.shape[1:])) 
    
    model.add(tf.keras.layers.Activation('sigmoid'))
        
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=default_model_configuration['learning_rate']
        )
    )
    return model