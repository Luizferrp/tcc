from .train import Epoch, Epoch_anal
from .ToImage import ToImage
from .preparation import preparation
from .fromCD2DL2Data import fromCD2DL2Data
from .device import device
from torchvision import transforms
import torch.optim as optim
from torch import nn
import numpy as np
import torch

def make(model: nn.Module, epocs: int, csv_path: str, batch_size: int = 128, xcol: int=1, lr: int=1e-3, transform: transforms=transforms.Compose([ ToImage() ])) -> tuple:
    
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_train, x_test, y_train, y_test = preparation(csv_path, xcol)

    data_test = fromCD2DL2Data(x_train, y_train, transform, batch_size)
    data_train = fromCD2DL2Data(x_test, y_test, transform, batch_size)
    all_metrics = {'treino':[], 'teste':[]}
    for epoch in range(1, epocs+1):
        metrics = Epoch(
            epoch=1,
            optimizer=optimizer,
            loss_fn=loss_fn,
            MODEL=model.train(),
            dataset=data_train,
            device=torch.device(device()),
            salvar_em="./resultados",
            class_names=["normal", "ataque"]
        )
        all_metrics['treino'].append(metrics)
        metrics = Epoch(
            epoch=1,
            optimizer=optimizer,
            loss_fn=loss_fn,
            MODEL=model.eval(),
            dataset=data_test,
            device=torch.device(device()),
            salvar_em="./resultados",
            class_names=["normal", "ataque"]
        )
        all_metrics['teste'].append(metrics)

        # register(, f'{out_path}-train--{epoch}')
        # register(, f'{out_path}-valid--{epoch}')
        #print(valid_m[epoch-1]["confusion_matrix"])
    
    print("Finished experiment!") 
    return all_metrics, model

