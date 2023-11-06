# Label generation functions
from importlib import import_module
import numpy as np
from scipy.stats import loguniform

# Legacy inclusions
from ...v1.utils import remove_singletons, normalize, random_hparam

def assert_indexing(indexing):
    indexing = indexing.lower()
    assert indexing in ('xy', 'ij'), \
        'Indexing method \'%s\' is not valid, must be (\'ij\' or \'xy\').' % indexing
    return indexing

class neg_loguniform():
    """
    A class that simply provides negative values corresponding to scipy's
    loguniform distribution. See `scipy.stats.loguniform` for details
    about initialization arguments.

    ONLY the `rvs` method should be used with this class. This is a shim.
    """
    def __init__(self, *args, **kwargs):
        self.loguniform = loguniform(*args, **kwargs)
    def rvs(self, *args, **kwargs):
        return -self.loguniform.rvs(*args, **kwargs)

class discretehp():
    """
    A class that may be sampled in a manner similar to scipy.stats distributions
    for random values in an array. This should be used with `synthimglbl` or
    `augment_hparam_dataset` to generate random parameters.
    """

    def __init__(self, values, seed=None):
        """
        Parameters
        ----------
        values : list-like
            A list of values from which to randomly sample.
        seed : int or None
            A random seed to use in seeding the random generator used internally
            by this class.
        """

        self._rng = np.random.default_rng(seed=seed)
        self._vals = values

    def rvs(self, size):
        """
        Randomly sample values from the internal array of this class.

        Parameters
        ----------
        size : int or tuple
            A desired size/shape of output values used for this function.
        
        Returns
        -------
        Random variates in the shape of `size`.
        """

        # log-uniform distribution, uniformly choose among log-distributed samples
        return self._rng.choice(a=self._vals,
                                size=size)

