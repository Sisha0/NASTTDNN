import torch
import numpy as np
import numpy.typing as npt
import typing as tp
from dataclasses import dataclass
from enum import Enum
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from nni.nas.evaluator.pytorch.lightning import DataLoader as LightningDataLoader


class TaskType(Enum):
    REGRESSION = 0
    CLASSIFICATION = 1


@dataclass
class TaskConfig:
    task_type: TaskType
    in_features: int
    out_features: int # either 1 for regression or num_classes for classification


def train_val_test_split(dataset_size: int) -> tp.Dict[str, npt.NDArray[np.int32]]:
    all_idx = np.arange(dataset_size, dtype=np.int32)
    trainval_idx, test_idx = train_test_split(all_idx, train_size=0.8)
    train_idx, val_idx = train_test_split(trainval_idx, train_size=0.8)
    return {'train': train_idx, 'val': val_idx, 'test': test_idx}


def get_data_loaders(features: npt.NDArray[np.float32], target: npt.NDArray[np.float32] | npt.NDArray[np.int32], batch_size: int, num_workers: int) -> tp.Dict[str, DataLoader]:
    loaders = {}

    for split_name in features:
        dataset = TensorDataset(torch.as_tensor(features[split_name]), torch.as_tensor(target[split_name]))
        shuffle = split_name == 'train'
        loader = LightningDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        loaders[split_name] = loader

    return loaders
