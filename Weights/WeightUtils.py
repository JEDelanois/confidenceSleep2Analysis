import numpy as np

# Reshape alwasy scares me but I think it is doing it correctly
# temp = np.array([[[0,1],[2,3],[4,5]],[[6,7],[8,9],[10,11]]]); print(temp); print(temp.reshape(2,6))
# weights shape (time, post, pre)
# retuns shape (time, unrolled weight index)
def unrollWeightsThroughTime(weights):
    return weights.reshape(-1, weights.shape[1] * weights.shape[2])