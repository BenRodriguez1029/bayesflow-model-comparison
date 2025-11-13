from keras import Model
from keras.layers import Dense
import bayesflow as bf
from keras.initializers import RandomNormal
import numpy as np
import keras.ops as ops


class EvidenceNetwork(Model):
    def __init__(self, output, **kwargs):
        super().__init__(**kwargs)

        # shared backbone network
        self.summary_network = bf.networks.DeepSet(summary_dim=8, dropout=None)
        self.classification_network = bf.networks.MLP(
            widths=[32] * 4, activation="silu", dropout=None)

        # output layer depends on output type
        if output == "p":
            self.output_layer = Dense(1, activation="sigmoid",
                                      kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))  # probability 0-1
        elif output == "K":
            self.output_layer = Dense(1, activation="softplus",
                                      kernel_initializer=RandomNormal(mean=np.log(np.exp(1)-1), stddev=0.01))  # strictly positive
        elif output == "log(K)":
            self.output_layer = Dense(1, activation=None,
                                      kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))  # unbounded
        else:
            raise ValueError("Invalid output type")

    def call(self, inputs, training=False):
        x, n = inputs
        x = self.summary_network(x, training=training)
        x = ops.concatenate([x, n], axis=-1)
        x = self.classification_network(x, training=training)
        return self.output_layer(x)


class MultiBayesEvidenceNetwork(Model):
    def __init__(self, num_models, **kwargs):
        super().__init__(**kwargs)

        # shared backbone network
        self.summary_network = bf.networks.DeepSet(summary_dim=8, dropout=None)
        self.classification_network = bf.networks.MLP(
            widths=[32] * 4, activation="silu", dropout=None)

        self.output_layer = Dense(num_models-1, activation=None,
                                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))  # unbounded

    def call(self, inputs, training=False):
        x, n = inputs
        x = self.summary_network(x, training=training)
        x = ops.concatenate([x, n], axis=-1)
        x = self.classification_network(x, training=training)
        return self.output_layer(x)


class BayesEvidenceNetwork(Model):
    def __init__(self, num_models, **kwargs):
        super().__init__(**kwargs)

        # shared backbone network
        self.summary_network = bf.networks.DeepSet(summary_dim=8, dropout=None)
        self.classification_network = bf.networks.MLP(
            widths=[32] * 4, activation="silu", dropout=None)

        self.output_layer = Dense(num_models-1, activation=None,
                                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))  # unbounded

    def call(self, x, training=False):
        x = self.summary_network(x, training=training)
        x = self.classification_network(x, training=training)
        return self.output_layer(x)
