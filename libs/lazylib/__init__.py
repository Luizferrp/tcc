from .now import now
from .make import make
from .device import device
from .validate import takeConfusionMatrix, takeRocCurve, register
from .models.AlexNet import AlexNet
from .models.AlexNetEE import AlexNetEE
from .models.GoogleLeNet import GoogLeNet
from .models.GoogleLeNetEE import GoogleLeNetEE


def takeFromDisk(path:str):
    import pickle
    recovered_object = []
    for i in range(1, 101):
        with open(path, 'rb') as f:
            recovered_object.append(pickle.load(f))
    return recovered_object