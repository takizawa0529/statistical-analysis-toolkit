import numpy as np

def Normalization(pos: [int]) -> float:
    if len(pos)<=1:
        raise ValueError("n must be greater than 2")
    return (pos - np.mean(pos, axis=0))/np.std(pos, axis=0)