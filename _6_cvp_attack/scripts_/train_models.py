from _6_cvp_attack_paper.scripts_.configuration_values import *

from _0_general_ML.data_utils.dataset_cards.taxibj import TaxiBJ

from _6_cvp_attack_paper._0_cav_detect.keras_model_with_call import Keras_Model_with_call



def main_sub(
    data_configuration,
    model_configuration
):
    
    my_data = TaxiBJ(
        dataset_folder=dataset_folder, 
        data_configuration=data_configuration
    )
    my_data.renew_data()
    my_data.info()
    
    my_model = Keras_Model_with_call(
        my_data, model_configuration,
        path=path
    )
    
    # model name
    model_name = 'model'
    for key in data_configuration.keys():
        model_name += '_({}-{})'.format(key, data_configuration[key])
    for key in model_configuration.keys():
        model_name += '_({}-{})'.format(key, model_configuration[key])
    print('Model name will be:', model_name)
        
    my_model.load_or_train(
        model_name, epochs=epochs, batch_size=batch_size,
        patience=15
    )
    
    return


def main():
    
    for history_length in taxibj_configuration['history_length']:
        
        for model_architecture in model_architectures:    
            if model_architecture=='tgcn':
                for message_dimension in model_configurations[model_architecture]['message_dimensions']:
                    for adjacent_nodes in model_configurations[model_architecture]['adjacent_nodes']:
                        
                        local_data_configuration = taxibj_configuration.copy()
                        local_data_configuration['history_length'] = history_length
                        
                        local_model_configuration = model_configurations[model_architecture].copy()
                        local_model_configuration['message_dimensions'] = message_dimension
                        local_model_configuration['adjacent_nodes'] = adjacent_nodes
                        
                        main_sub(
                            local_data_configuration,
                            local_model_configuration
                        )
                    
            else:
                for hidden_layers in model_configurations[model_architecture]['hidden_layers']:
                    
                    local_data_configuration = taxibj_configuration.copy()
                    local_data_configuration['history_length'] = history_length
                    
                    local_model_configuration = model_configurations[model_architecture].copy()
                    local_model_configuration['hidden_layers'] = hidden_layers
                    
                    main_sub(
                        local_data_configuration,
                        local_model_configuration
                    )
                
    return

