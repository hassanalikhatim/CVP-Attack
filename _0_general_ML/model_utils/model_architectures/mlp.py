import tensorflow as tf


from _0_general_ML.data_utils.dataset import Dataset



def mlp_baseline(
    data: Dataset, model_configuration: dict,
    **kwargs
):
    
    default_model_configuration = {
        'hidden_layers': 1,
        'weight_decay': None,
        'learning_rate': 1e-4
    }
    for key in model_configuration.keys():
        default_model_configuration[key] = model_configuration[key]
    
    in_sample, out_sample = data.x_train[:1], data.y_train[:1]
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=in_sample.shape[1:]))
    
    for k in range(default_model_configuration['hidden_layers']):
        model.add(
            tf.keras.layers.Dense(
                512, activation='relu', 
                kernel_regularizer=default_model_configuration['weight_decay']
            )
        )
    
    model.add(
        tf.keras.layers.Dense(
            out_sample[0].reshape(-1).shape[0], 
            kernel_regularizer=default_model_configuration['weight_decay']
        )
    )
    model.add(tf.keras.layers.Reshape(target_shape=out_sample.shape[1:])) 
    
    model.add(tf.keras.layers.Activation('sigmoid'))
        
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=default_model_configuration['learning_rate']
        )
    )
    
    return model