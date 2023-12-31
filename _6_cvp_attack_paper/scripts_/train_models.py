from _6_cvp_attack_paper.scripts_.configuration_values import *

from _0_general_ML.data_utils.dataset_cards.taxibj import TaxiBJ

from _0_general_ML.model_utils.model import Keras_Model



def main_sub(
    data_configuration,
    model_configuration
):
    
    my_data = TaxiBJ(
        dataset_folder=dataset_folder, 
        data_configuration=data_configuration
    )
    my_data.renew_data()
    
    my_model = Keras_Model(
        my_data, model_configuration,
        path=path
    )
    
    # model name
    model_name = 'model'
    for key in data_configuration.keys():
        model_name += '_({}-{})'.format(key, data_configuration[key])
    for key in model_configuration.keys():
        model_name += '_({}-{})'.format(key, model_configuration[key])
        
    my_model.load_or_train(
        model_name, epochs=epochs, 
        batch_size=batch_size
    )
    
    return


def main():
    
    for m_arc, model_architecture in model_architectures:
        
        if model_architecture=='tgcn':
            for m, message_dimension in model_configurations[model_architecture]['message_dimensions']:
                for d_A, adjacent_nodes in model_configurations[model_architecture]['num_neighbours']:
                    
                    local_data_configuration = taxibj_configuration
                    
                    local_model_configuration = {
                        'model_architecture': 'tgcn',
                        'hidden_layers': 1,
                        'message_dimensions': message_dimension,
                        'adjacent_nodes': adjacent_nodes
                    }
                    for key in model_configurations[model_architecture].keys():
                        if key not in local_model_configuration.keys():
                            local_model_configuration[key] = model_configurations[model_architecture][key]
                    
                    main_sub(
                        local_data_configuration,
                        local_model_configuration
                    )
                
        else:
            
            for l, hidden_layers in model_configurations[model_architecture]['model_depths']:
                
                local_data_configuration = taxibj_configuration
                    
                local_model_configuration = {
                    'model_architecture': 'tgcn',
                    'hidden_layers': hidden_layers
                }
                for key in model_configurations[model_architecture].keys():
                    if key not in local_model_configuration.keys():
                        local_model_configuration[key] = model_configurations[model_architecture][key]
                
                main_sub(
                    local_data_configuration,
                    local_model_configuration
                )
                
    return

