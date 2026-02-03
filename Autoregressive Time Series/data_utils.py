import numpy as np
from keras.utils import Sequence


class DataSimulator(Sequence):
    def __init__(self, batches_per_epoch, simulator, batch_size=32, normalize=True, **kwargs):
        super().__init__(**kwargs)
        self.batches_per_epoch = batches_per_epoch
        self.simulator = simulator
        self.batch_size = batch_size
        self.normalize = normalize

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, index):
        X_list, y_list = self.simulator(self.batch_size)

        X_batch = np.array(X_list)
        y_batch = np.array(y_list)

        if self.normalize:
            # mean = np.mean(X_batch, axis=1, keepdims=True)
            # std = np.std(X_batch, axis=1, keepdims=True)
            # X_batch = (X_batch - mean) / (std + 1e-8)

            X_batch = X_batch * 50.0

        perm = np.random.permutation(self.batch_size)
        X_batch = X_batch[perm]
        y_batch = y_batch[perm]

        X_batch = X_batch[..., np.newaxis]
        y_batch = y_batch[..., np.newaxis]

        return X_batch, y_batch
