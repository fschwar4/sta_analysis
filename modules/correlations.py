import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy.stats import pearsonr

def calc_mse(array1, array2):
    """
    Calculates the mean square error of 2 n-dimensional arrays of the same shape.

    Note:
        Values close to 0 hint at similar images (or at least little deviations) and therefore a better correlation.
    """
    if array1.shape != array2.shape:
        raise ValueError(f'Shapes must be identical to compare: {array1.shape} vs {array2.shape}')

    return np.mean(np.square(array1 - array2))

def calc_pearson(array1, array2):
    statistic, p = pearsonr(array1.ravel(), array2.ravel())
    return statistic, p

def calc_cosine_similarity(array1, array2):
    if array1.shape != array2.shape:
        raise ValueError(f'Shapes must be identical to compare: {array1.shape} vs {array2.shape}')
    
    if norm(array1) == 0 or norm(array2) == 0:
        raise ValueError(f'One given array (image) has norm 0.')

    array1 = array1.ravel()
    array2 = array2.ravel()
    cos_sim = dot(array1, array2) / (norm(array1) * norm(array2))
    return cos_sim