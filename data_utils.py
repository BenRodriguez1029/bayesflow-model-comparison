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


def sample_beta(s, n, alpha=1.0, beta=1.0):
    return np.random.beta(alpha, beta, size=(s, n))


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


class DataGeneratorMulti(Sequence):
    def __init__(self, batches_per_epoch, sample_funcs, n_generator, batch_size=32, n_norm=30, **kwargs):
        super().__init__(**kwargs)
        self.batches_per_epoch = batches_per_epoch
        self.sample_funcs = sample_funcs
        self.n_generator = n_generator
        self.batch_size = batch_size
        self.n_models = len(sample_funcs)
        self.n_norm = n_norm

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, index):
        sample_n = self.n_generator()

        base_size = self.batch_size // self.n_models
        sizes = [base_size] * self.n_models
        for i in range(self.batch_size % self.n_models):
            sizes[i] += 1

        X_list = []
        y_list = []
        for model_idx, (func, size) in enumerate(zip(self.sample_funcs, sizes)):
            X_model = func(size, sample_n)
            X_list.append(X_model)
            y_list.append(np.full(size, model_idx, dtype=np.int32))

        X_batch = np.concatenate(X_list, axis=0)
        y_batch = np.concatenate(y_list, axis=0)

        perm = np.random.permutation(self.batch_size)
        X_batch = X_batch[perm]
        y_batch = y_batch[perm]

        n_batch = np.full((self.batch_size, 1), sample_n /
                          self.n_norm, dtype=np.float32)

        return (X_batch[:, :, None], n_batch), y_batch[:, None]


class DataSimulator(Sequence):
    def __init__(self, batches_per_epoch, simulators, batch_size=32, **kwargs):
        super().__init__(**kwargs)
        self.batches_per_epoch = batches_per_epoch
        self.simulators = simulators
        self.batch_size = batch_size
        self.n_models = len(simulators)

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, index):
        base_size = self.batch_size // self.n_models
        sizes = [base_size] * self.n_models
        for i in range(self.batch_size % self.n_models):
            sizes[i] += 1

        X_list = []
        y_list = []
        for model_idx, (func, size) in enumerate(zip(self.simulators, sizes)):
            X_model = np.array(func(size))
            X_list.append(X_model)
            y_list.append(np.full(size, model_idx, dtype=np.int32))

        X_batch = np.concatenate(X_list, axis=0)
        y_batch = np.concatenate(y_list, axis=0)

        perm = np.random.permutation(self.batch_size)
        X_batch = X_batch[perm]
        y_batch = y_batch[perm]

        return X_batch[:, :, None], y_batch[:, None]
