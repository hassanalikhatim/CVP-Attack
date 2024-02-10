import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
gpus = tf.config.list_physical_devices('GPU')
try:
    tf.config.set_visible_devices(gpus[3], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


from _6_cvp_attack_paper.scripts_.train_models import main as train_models
from _6_cvp_attack_paper.scripts_.non_adaptive_attack_evaluation import main as non_adaptive_attack_evaluation
from _6_cvp_attack_paper.scripts_.cav_detect_evaluation import main as cav_detect_evaluation
from _6_cvp_attack_paper.scripts_.adaptive_attack_evaluation import main as adaptive_attack_evaluation
from _6_cvp_attack_paper.scripts_.physical_attack_evaluation import main as physical_attack_evaluation



if __name__ == '__main__':
    
    train_models()
    non_adaptive_attack_evaluation()
    cav_detect_evaluation()
    adaptive_attack_evaluation()
    physical_attack_evaluation()
    