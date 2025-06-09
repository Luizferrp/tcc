from .train import Epoch
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

def make(model: nn.Module, epocs: int, batch_size: int, csv_path: str, xcol: int=1, lr: int=1e-3, transform: transforms=transforms.Compose([ ToImage() ])) -> tuple:
    
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_train, x_test, y_train, y_test = preparation(csv_path, xcol)

    data_test = fromCD2DL2Data(x_train, y_train, transform, 128)
    data_train = fromCD2DL2Data(x_test, y_test, transform, 128)
    ttmetrics = {'treino':[], 'teste':[]}
    for epoch in range(1, epocs+1):
        metrics = Epoch(
            epoch=1,
            optimizer=optimizer,
            loss_fn=loss_fn,
            MODEL=model.train(),
            dataset=data_train,
            device=torch.device('cpu'),
            salvar_em="./resultados",
            class_names=["normal", "ataque"]
        )
        ttmetrics['treino'].append(metrics)
        metrics = Epoch(
            epoch=1,
            optimizer=optimizer,
            loss_fn=loss_fn,
            MODEL=model.eval(),
            dataset=data_test,
            device=torch.device('cpu'),
            salvar_em="./resultados",
            class_names=["normal", "ataque"]
        )
        ttmetrics['teste'].append(metrics)


        # register(, f'{out_path}-train--{epoch}')
        # register(, f'{out_path}-valid--{epoch}')
        #print(valid_m[epoch-1]["confusion_matrix"])
    
    print("Finished experiment!") 
    return MODEL

