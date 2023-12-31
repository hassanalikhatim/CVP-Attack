import tensorflow as tf


from .custom_layers.st_resnet_layers import STResNet_Unit, Fusion



def st_resnet_model_function(
    model_in, 
    repeats=12, 
    regularizer=None, 
    multiple_input=True
):
    
    inter0 = tf.keras.layers.Conv2D(
        64, (7,7), padding='same', activation='relu', 
        kernel_regularizer=regularizer
    )(model_in[:,0])
    
    if multiple_input:
        inter1 = tf.keras.layers.Conv2D(
            64, (7,7), padding='same', activation='relu', 
            kernel_regularizer=regularizer
        )(model_in[:,1])
        inter2 = tf.keras.layers.Conv2D(
            64, (7,7), padding='same', activation='relu', 
            kernel_regularizer=regularizer
        )(model_in[:,2])
    
    for i in range(repeats):
        inter0 = STResNet_Unit(64, (7,7), kernel_regularizer=regularizer)(inter0)
        if multiple_input:
            inter1 = STResNet_Unit(64, (7,7), kernel_regularizer=regularizer)(inter1)
            inter2 = STResNet_Unit(64, (7,7), kernel_regularizer=regularizer)(inter2)
    
    inter0 = tf.keras.layers.Conv2D(
        10, (7,7), padding='same', activation='relu', 
        kernel_regularizer=regularizer
    )(inter0)
    
    if multiple_input:
        inter1 = tf.keras.layers.Conv2D(
            10, (7,7), padding='same', activation='relu', 
            kernel_regularizer=regularizer
        )(inter1)
        inter2 = tf.keras.layers.Conv2D(
            10, (7,7), padding='same', activation='relu', 
            kernel_regularizer=regularizer
        )(inter2)
    
        inter0 = Fusion(kernel_regularizer=regularizer)([inter0, inter1, inter2])
    
    return inter0
    
    
def st_resnet_baseline(
    data, model_configuration: dict,
    **kwargs
):
    
    default_model_configuration = {
        'hidden_layers': [1],
        'weight_decay': None,
        'learning_rate': 1e-4
    }
    for key in model_configuration.keys():
        default_model_configuration[key] = model_configuration[key]
    
    in_samples, out_samples = data.x_train[:1], data.y_train[:1]
    
    model_in = tf.keras.layers.Input(shape=in_samples.shape[1:])
    
    model_out = st_resnet_model_function(
        model_in, repeats=default_model_configuration['hidden_layers'], 
        regularizer=default_model_configuration['weight_decay'], 
        multiple_input=data.data_configuration['multiple_input']
    )
    
    model_out = tf.keras.layers.Flatten()(model_out)
    model_out = tf.keras.layers.Dense(out_samples[0].reshape(-1).shape[0])(model_out)
    model_out = tf.keras.layers.Reshape(target_shape=out_samples.shape[1:])(model_out)
    
    model_out = tf.keras.layers.Activation('sigmoid')(model_out)
    
    st_resnet = tf.keras.models.Model(inputs=model_in, outputs=model_out)
    
    st_resnet.compile(
        loss='mse', 
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=default_model_configuration['learning_rate']
        )
    )
    
    return st_resnet

