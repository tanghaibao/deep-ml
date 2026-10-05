import numpy as np

def rotation_layer(X, angle):
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.array(X) @ R.T