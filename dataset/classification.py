import numpy as np
import numpy.typing as npt
import typing as tp
from sklearn.preprocessing import QuantileTransformer
from sklearn.datasets import fetch_covtype, fetch_openml
from torch.utils.data import DataLoader
from .common import TaskType, TaskConfig, train_val_test_split, get_data_loaders


def preprocess_features(data: npt.NDArray[np.float32], split: tp.Dict[str, npt.NDArray[np.int32]]):
    train_data = data[split['train']]
    noise = (
        np.random.default_rng(0)
        .normal(0.0, 1e-5, train_data.shape)
        .astype(train_data.dtype)
    )
    preprocessing = QuantileTransformer(
        n_quantiles=max(min(len(train_data) // 30, 1000), 10),
        output_distribution="normal",
        subsample=10**9,
    ).fit(train_data + noise)

    return {split_name: preprocessing.transform(data[split_idx]) for split_name, split_idx in split.items()}


def preprocess_target(target: npt.NDArray[np.float32], split: tp.Dict[str, npt.NDArray[np.int32]]):
    return {split_name: target[split_idx] for split_name, split_idx in split.items()} # subtracting 1 to make classes be in [0, num_classes) range


def fetch_data(name: tp.Literal['co', 'ja'], batch_size: int = 256, num_workers: int = 4) -> tp.Tuple[TaskConfig, tp.Dict[str, DataLoader]]:
    if name == 'co':
        dataset = fetch_covtype()
        data, target = dataset['data'], dataset['target']
    else:
        dataset = fetch_openml(name='jannis', version=1)
        data, target = dataset['data'].to_numpy(), dataset['target'].to_numpy()

    n_samples, n_features = data.shape
    n_classes = np.unique(target).shape[0]

    split = train_val_test_split(n_samples)
    preprocessed_features = preprocess_features(data.astype(np.float32), split)
    preprocessed_target = preprocess_target(target.astype(np.int64), split)

    loaders = get_data_loaders(preprocessed_features, preprocessed_target, batch_size=batch_size, num_workers=num_workers)
    return TaskConfig(TaskType.CLASSIFICATION, n_features, n_classes), loaders
