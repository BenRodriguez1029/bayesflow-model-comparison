import numpy as np
from keras.utils import Sequence

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

        return X_batch, y_batch[:, None]
