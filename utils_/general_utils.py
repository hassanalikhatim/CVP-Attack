import os
import numpy as np



def confirm_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def neighbour_image_pixels(
    n, h, w, 
    num_neighbour=1
):
    
    y = n%w
    x = int(n/w)
    
    _x_values = np.arange(np.max([x-num_neighbour, 0]), np.min([x+num_neighbour+1, h]))
    _y_values = np.arange(np.max([y-num_neighbour, 0]), np.min([y+num_neighbour+1, w]))
    
    x_values = []
    for value in _x_values:
        x_values += [value]*len(_y_values)
    y_values = list(_y_values)*len(_x_values)
    
    return np.array(x_values)*w + np.array(y_values)


def prepare_adjacency_matrix(in_sample, adjacent_nodes):
    
    # preparing the adjacency matrix
    (h, w) = in_sample[0,:,:,0].shape
    
    adjacency_matrix = np.zeros(
        (h*w, h*w)
    ).astype('float32')
    for k in np.arange(h*w):
        neighbours = neighbour_image_pixels(
            k, h, w, 
            num_neighbour=adjacent_nodes
        )
        adjacency_matrix[k, neighbours] = 1.
    
    return adjacency_matrix