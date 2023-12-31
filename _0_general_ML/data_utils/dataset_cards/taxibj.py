import numpy as np


from ..dataset import Dataset

from utils_.general_utils import confirm_directory, neighbour_image_pixels



class TaxiBJ(Dataset):
    
    def __init__(
        self,
        dataset_folder='../../_Datasets/',
        test_ratio=0.2, 
        data_configuration={},
        **kwargs
    ):
        
        super().__init__(
            data_name='TaxiBJ',
            preferred_size=-1
        )
        
        self.dataset_folder = dataset_folder
        self.test_ratio = test_ratio
        
        self.preapre_data_configuration(data_configuration)
        
        return
    
    
    def preapre_data_configuration(self, data_configuration: dict):
        
        self.data_configuration = {
            'year': 16,
            'history_length': 5,
            'multiple_input': False,
            'multiple_output': True
        }
        for key in data_configuration.keys():
            self.data_configuration[key] = data_configuration[key]
        
        return
    
    
    def prepare_data(self):
        
        data = np.load(
            self.dataset_folder + 'TaxiBJ/TaxiBJ{}.npy'.format(self.data_configuration['year'])
        )
        print("Data shape: ", data.shape, np.max(data), np.min(data))
        
        if self.data_configuration['multiple_input']:
            base_x, base_y = self.generate_resnet_data(data)
        else:
            base_x, base_y = self.generate_simple_data(data)
        
        base_x, base_y = self.shuffle(base_x, base_y)
        
        print(base_x.shape, base_y.shape)
        
        x_train = base_x[:-int(self.test_ratio*len(base_x))]
        y_train = base_y[:-int(self.test_ratio*len(base_x))]
        x_test = base_x[-int(self.test_ratio*len(base_x)):]
        y_test = base_y[-int(self.test_ratio*len(base_x)):]
        
        x_train = np.clip(x_train/1000, 0, 1).astype(np.float32)
        y_train = np.clip(y_train/1000, 0, 1).astype(np.float32)
        x_test = np.clip(x_test/1000, 0, 1).astype(np.float32)
        y_test = np.clip(y_test/1000, 0, 1).astype(np.float32)
        
        return x_train, y_train, x_test, y_test
    
    
    def generate_simple_data(
        self, base_data
    ):
        
        history = self.data_configuration['history_length']
        multiple_output = self.data_configuration['multiple_output']
        
        data_x, data_y = [], []
        for k in np.arange(history, len(base_data)):
            closeness = np.append(
                base_data[k+np.arange(-history, 0).astype('int')][:,0], 
                base_data[k+np.arange(-history, 0).astype('int')][:,1],
                axis=0
            )
            
            data_x.append([closeness])
            if multiple_output:
                data_y.append(np.transpose(base_data[k], (1,2,0)))
            else:
                data_y.append(np.expand_dims(np.mean(base_data[k], axis=0), axis=2))
            
        return np.transpose(np.array(data_x), (0,1,3,4,2)), np.array(data_y)
    
    
    def generate_resnet_data(
        self, base_data
    ):
        
        history = self.data_configuration['history_length']
        
        data_x, data_y = [], []
        for k in np.arange(history*2*24*7, len(base_data)):
            
            trend = np.append(
                base_data[k+np.arange(-history, 0).astype('int')*2*24*7][:,0], 
                base_data[k+np.arange(-history, 0).astype('int')*2*24*7][:,1], 
                axis=0
            )
            period = np.append(
                base_data[k+np.arange(-history, 0).astype('int')*2*24][:,0], 
                base_data[k+np.arange(-history, 0).astype('int')*2*24][:,1], 
                axis=0
            )
            closeness = np.append(
                base_data[k+np.arange(-history, 0).astype('int')*2][:,0], 
                base_data[k+np.arange(-history, 0).astype('int')*2][:,1], 
                axis=0
            )
            
            data_y.append(
                np.expand_dims(
                    np.mean(base_data[k], axis=0), 
                    axis=2
                )
            )
            data_x.append([closeness, period, trend])
        
        return np.transpose(np.array(data_x), (0,1,3,4,2)), np.array(data_y)
    
    
    def shuffle(self, base_x, base_y):
        
        confirm_directory(self.dataset_folder+'shuffled_indices/')
        
        indices_filename = self.data_name + str(self.data_configuration['year'])
        indices_filename += '_h' + str(self.data_configuration['history_length'])
        
        try:
            shuffled_indices = np.load(self.path+'shuffled_indices/'+indices_filename+'.npy')
            print("Shuffled indices loaded.")
        
        except:
            from sklearn.utils import shuffle
            
            shuffled_indices = shuffle(
                np.arange( int(self.test_ratio * len(base_x)) )
            )
            shuffled_indices = np.append(
                shuffled_indices, 
                np.arange(
                    int( self.test_ratio*len(base_x) ), 
                    len(base_x)
                ), 
                axis=0
            )
            
            assert len(shuffled_indices) == len(base_x), 'After shuffling the data size should remain same.'
            
            confirm_directory(self.dataset_folder + 'shuffled_indices/')
            np.save(
                self.dataset_folder + 'shuffled_indices/' + indices_filename + '.npy', 
                shuffled_indices
            )
        
        base_x, base_y = base_x[shuffled_indices], base_y[shuffled_indices]
        
        return base_x, base_y
    
    
    def info(self):
        print('Min: {} | Max: {}'.format( np.min(self.x_train), np.max(self.x_train) ))
    
    
    def _prepare_convolutional_adjacency_matrix(self):
        
        (h, w) = self.x[0,0,:,:,0].shape
        
        self.adjacency_matrix = np.zeros((h*w, h*w)).astype('float32')
        for k in np.arange(h*w):
            neighbours = neighbour_image_pixels(k, h, w, num_neighbour=self.num_neighbour)
            self.adjacency_matrix[k, neighbours] = 1.
        
        self.adjacency_matrix_type = '('+str(self.num_neighbour)+')'
        
        return
    
    