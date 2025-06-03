import torch
from functools import cache

class local_device():

    def __init__(self):
        if not self.device:
            self.devide = "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
    
    def _device(self):
        return self._device
    
    def __str__(self):
        return self._device()
    
    def __repr__(self):
        return self._device()
    
    def __call__(self):
        return self._device()
    
    def __del__(self):
        del self._device()
        return None
    
device = local_device()