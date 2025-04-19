from .train import train
from .ToImage import ToImage
from .preparation import preparation
from .fromCD2DL2Data import fromCD2DL2Data
from .validate import validate, register
from torchvision import transforms
import torch.optim as optim
from torch import nn
import numpy as np
import random
import torch

def transformate(t: transforms = ToImage):
    return transforms.Compose([
        t(),
    ])

def make(EPOCHS: int, BATCH_SIZE: int, LEARNING_RATE: int, csv_path: str, out_path:str, MODEL: nn.Module, SEED:int = 1701, xcol: int=1, transform: transforms=transformate()) -> tuple:
    
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)

    x_train, x_test, y_train, y_test = preparation(csv_path, xcol)

    data_test = fromCD2DL2Data(x_train, y_train, transform, 128)
    data_train = fromCD2DL2Data(x_test, y_test, transform, 128)
    
    for epoch in range(1, EPOCHS+1):
        register(train(epoch, optimizer, loss_fn, MODEL, data_train), f'{out_path}-train--{epoch}')
        register(validate(epoch, loss_fn, MODEL, data_test), f'{out_path}-valid--{epoch}')
        #print(valid_m[epoch-1]["confusion_matrix"])
    
    print("Finished experiment!") 
    return MODEL

class mmodel:
    def __init__(self, name):
        self.name = name
    
    def data(path_data:str, buffer:int) -> None:
        pass

    def buffer(path_out:str) -> None:
        pass

    def train(epoch:str) -> None:
        pass

    def validate(epoch:str) -> None:
        pass

def make2(model:object, epoch:int, buffer:int, path_data:str, path_out:str='output', seed:int = 91) -> dict:

    np.random.seed(seed)
    torch.manual_seed(seed)

    model.data(path_data, buffer)
    model.buffer(path_out)

    for epoch in range(1, epoch):
        model.train(epoch)
        model.validate(epoch)

    print('finish')
    return model
    
