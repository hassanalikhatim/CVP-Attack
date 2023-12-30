import os



def confirm_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return