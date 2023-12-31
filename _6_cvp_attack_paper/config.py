# dataset configurations
taxibj_configuration = {
    'year': 16,
    'history_lengths': [3],
    'multiple_input': False,
    'multiple_output': True
}


# model configurations
st_resnet_config = {
    'model_architecture': 'st_resnet', 
    'model_depths': [1, 2, 3],
    'weight_decay': None,
    'learning_rate': 1e-4
}
mlp_config = {
    'model_architecture': 'mlp', 
    'model_depths': [3, 5, 10],
    'weight_decay': None,
    'learning_rate': 1e-4
}
tgcn_config = {
    'model_architecture': 'tgcn', 
    'model_depths': [1],
    'weight_decay': None,
    'message_dimensions': [1, 3, 5, 10],
    'num_neighbours': [1, 3, 5, 10],
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
    'epsilon_per_iteration': 0.01
}
upa_configuration = {
    'name': 'upa',
    'perturbation_budgets': [0.01, 0.03, 0.05, 0.07, 0.1],
    'epsilon_per_iteration': 0.01
}

attack_configurations = {
    'fgsm': fgsm_configuration,
    'ifgsm': ifgsm_configuration,
    'pgd': pgd_configuration,
    'upa': upa_configuration
}


# changeable configurations
# general
path = 'all_results/test_results/'

# dataset
dataset_folder = '../../_Datasets/'

# model
model_architectures = ['mlp']
batch_size = 64
epochs = 500

# attack
attack_names = ['fgsm']