# dataset configurations
taxibj_configuration = {
    'year': 16,
    'history_length': [3, 5, 10, 15, 20],
    'multiple_input': False,
    'multiple_output': True
}


# model configurations
st_resnet_config = {
    'model_architecture': 'st_resnet', 
    'hidden_layers': [1, 2, 3],
    'weight_decay': None,
    'learning_rate': 1e-4
}
mlp_config = {
    'model_architecture': 'mlp', 
    'hidden_layers': [3, 5, 10],
    'weight_decay': None,
    'learning_rate': 1e-4
}
tgcn_config = {
    'model_architecture': 'tgcn', 
    'hidden_layers': 1,
    'weight_decay': None,
    'message_dimensions': [1, 3, 5, 10],
    'adjacent_nodes': [1, 3, 5, 10],
    'learning_rate': 1e-4
}

model_configurations = {
    'st_resnet': st_resnet_config,
    'mlp': mlp_config,
    'tgcn': tgcn_config
}


# attack configurations
fgsm_configuration = {
    'name': 'fgsm',
    'perturbation_budgets': [0.01, 0.03, 0.05, 0.07, 0.1]
}
ifgsm_configuration = {
    'name': 'ifgsm',
    'perturbation_budgets': [0.01, 0.03, 0.05, 0.07, 0.1]
}
pgd_configuration = {
    'name': 'pgd',
    'perturbation_budgets': [0.01, 0.03, 0.05, 0.07, 0.1],
    'device_budgets': [500, 1000, 5000, 10000, 15000],
    'epsilon_per_iteration': 0.01
}
cvp_configuration = {
    'name': 'cvp',
    'perturbation_budgets': [0.01, 0.03, 0.05, 0.07, 0.1],
    'device_budgets': [500, 1000, 5000, 10000, 15000],
    'epsilon_per_iteration': 0.01
}

attack_configurations = {
    'fgsm': fgsm_configuration,
    'ifgsm': ifgsm_configuration,
    'pgd': pgd_configuration,
    'cvp': cvp_configuration
}


# changeable configurations
# general
path = '__all_results__/test_results/'

# dataset
dataset_folder = '../../_Datasets/'

# model
model_architectures = ['mlp', 'st_resnet', 'tgcn']
batch_size = 64
epochs = 100

# attack
attack_names = ['fgsm', 'ifgsm', 'pgd', 'cvp']
n_targets = 300
iterations = 500