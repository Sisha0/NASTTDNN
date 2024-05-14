import numpy as np
import numpy.typing as npt
import typing as tp
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.datasets import fetch_california_housing
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


def preprocess_target(data: npt.NDArray[np.float32], split: tp.Dict[str, npt.NDArray[np.int32]]):
    scaler = StandardScaler().fit(data[split['train']])
    return {split_name: scaler.transform(data[split_idx]) for split_name, split_idx in split.items()}


def fetch_data(batch_size: int = 256, num_workers: int = 4) -> tp.Tuple[TaskConfig, tp.Dict[str, DataLoader]]:
    dataset = fetch_california_housing()
    n_samples, n_features = dataset['data'].shape

    split = train_val_test_split(n_samples)
    preprocessed_features = preprocess_features(dataset['data'].astype(np.float32), split)
    preprocessed_target = preprocess_target(dataset['target'].astype(np.float32)[:, None], split)

    loaders = get_data_loaders(preprocessed_features, preprocessed_target, batch_size=batch_size, num_workers=num_workers)
    return TaskConfig(TaskType.REGRESSION, n_features, 1), loaders
