import numpy as np
import random
from keras.utils import Sequence

# --- Simulator Functions ---


def prior_alternative():
    return np.random.normal(loc=0, scale=1)


def sample_model_0(sample_size, n=30):
    samples = np.random.normal(loc=0, scale=1, size=(sample_size, n))
    return samples


def sample_model_1(sample_size, n=30):
    mus = np.array([prior_alternative() for _ in range(sample_size)])
    samples = np.random.normal(
        loc=mus[:, None], scale=1, size=(sample_size, n))
    return samples

# --- n Generator Functions ---


def variable_n(min_val=20, max_val=30):
    return random.randint(min_val, max_val)


def constant_n(n=30):
    return n

# --- DataGenerator Class ---


class DataGenerator(Sequence):
    def __init__(self, batches_per_epoch, n_generator, batch_size=32, n_norm=30, **kwargs):
        super().__init__(**kwargs)
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.n_gen = n_generator
        self.n_norm = n_norm

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, index):
        sample_n = self.n_gen()
        batch_size_0 = self.batch_size // 2
        batch_size_1 = self.batch_size - batch_size_0

        data_0 = sample_model_0(batch_size_0, n=sample_n)
        data_1 = sample_model_1(batch_size_1, n=sample_n)

        X_batch = np.concatenate([data_0, data_1], axis=0)
        y_batch = np.concatenate(
            [np.zeros(batch_size_0), np.ones(batch_size_1)], axis=0)

        perm = np.random.permutation(len(X_batch))
        X_batch = X_batch[perm]
        y_batch = y_batch[perm]

        n_batch = np.full((self.batch_size, 1),
                          (sample_n / self.n_norm), dtype=np.float32)

        return (X_batch[:, :, None], n_batch), y_batch[:, None]
