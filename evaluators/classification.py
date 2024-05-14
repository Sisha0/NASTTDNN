import nni
from torch.optim import Optimizer, Adam
from torch import nn
from torch.utils.data import DataLoader
from nni.nas.evaluator.pytorch.lightning import Classification


@nni.trace
class ClassificationEvaluator(Classification):
    def __init__(self,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: Optimizer = Adam,
                 train_dataloaders: DataLoader | None = None,
                 val_dataloaders: DataLoader | None = None,
                 num_classes: int = 2,
                 max_epochs: int = 1
                ):
        super().__init__(criterion=nn.CrossEntropyLoss,
                         learning_rate=learning_rate,
                         weight_decay=weight_decay,
                         optimizer=optimizer,
                         train_dataloaders=train_dataloaders,
                         val_dataloaders=val_dataloaders,
                         num_classes=num_classes,
                         max_epochs=max_epochs
                        )

