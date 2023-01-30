from functools import cached_property
import numpy as np


class Kernel:
    def __init__(self, n_base_fn=256, spline_order=2, a=1e-5, b=1e-5):
        self.n_base_fn = n_base_fn
        self.betas = np.arange(n_base_fn)
        self.spline_order = spline_order
        self.a = a
        self.b = b

    def feature_map(self, data):
        _cos = np.cos(self.betas[:,None] @ data)
        _weights = (2/np.pi) ** (1/2) * np.concatenate((np.ones(1) / np.sqrt(2), np.ones(self.n_base_fn-1)))
        return _weights[:,None] * _cos

    @cached_property
    def feature_weights(self):
        return 1/(self.a * (self.betas**(2*self.spline_order)) + self.b)
