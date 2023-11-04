import keras.metrics
import numpy as np
from config_parser import Parser

import tensorflow as tf
from keras.layers import Input, LSTM, Conv2D, Conv1D, Dense, Dropout, GlobalMaxPool1D, Activation
from keras.models import Model
from tensorflow.math import l2_normalize
from tensorflow.keras.losses import cosine_similarity


class simCLR(Model):
    def __init__(self, inputs_shape: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.contrastive_loss_tracker = None
        self.contrastive_optimizer = None

        config = Parser()
        config.get_args()
        self.conf = config

        self.inputs_shape = inputs_shape
        self.L = 96
        self.encoder = self.get_encoder(model_name='Encoder')
        if self.conf.attach_head:
            self.encoder = self.attach_head()

        self.batch_size = self.conf.batch_size
        self.temperature = self.conf.clr_temp
        self.negative_mask = tf.cast(~tf.eye(self.batch_size * 2, self.batch_size * 2, dtype=bool), tf.float32)

    def summary(
        self,
        line_length=None,
        positions=None,
        print_fn=None,
        expand_nested=False,
        show_trainable=False,
        layer_range=None,
    ):
        self.encoder.summary()

    def compile(
        self,
        optimizer: tf.optimizers.Optimizer = "rmsprop",
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        jit_compile=None,
        **kwargs,
    ):
        super().compile(**kwargs)

        self.contrastive_optimizer = optimizer
        self.contrastive_loss_tracker = keras.metrics.Mean(name='c_loss')

    def get_encoder(self, model_name: str = 'base_model') -> Model:
        inputs = Input(shape=self.input_shape, name='input')
        x = inputs
        x = Conv1D(
            32, 24,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4)
        )(x)
        x = Dropout(0.1)(x)

        x = tf.keras.layers.Conv1D(
            64, 16,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
        )(x)
        x = Dropout(0.1)(x)

        x = tf.keras.layers.Conv1D(
            self.L, 8,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
        )(x)
        x = Dropout(0.1)(x)

        x = GlobalMaxPool1D(data_format='channels_last', name='global_max_pooling1d')(x)

        return Model(inputs, x, name=model_name)

    def attach_head(self, hidden_1=256, hidden_2=128, hidden_3=50,
                    name: str = 'Encoder with Head') -> Model:
        inputs = self.encoder.input
        x = self.encoder.output

        projection_1 = Dense(hidden_1)(x)
        projection_1 = Activation("relu")(projection_1)
        projection_2 = Dense(hidden_2)(projection_1)
        projection_2 = Activation("relu")(projection_2)
        projection_3 = Dense(hidden_3)(projection_2)

        simclr_model = Model(inputs, projection_3, name=name)

        return simclr_model

    def get_contrastive_loss(self, anchor_embeddings, target_embeddings, how: str = 'fast') -> tf.float32:
        anchor_embeddings = l2_normalize(anchor_embeddings, axis=1)
        target_embeddings = l2_normalize(target_embeddings, axis=1)

        representations = tf.concat([anchor_embeddings, target_embeddings], axis=0)

        sim_matrix = -cosine_similarity(tf.expand_dims(representations, axis=1),
                                        tf.expand_dims(representations, axis=0),
                                        axis=2)

        if how == 'naive':
            def get_loss_ij(i, j):
                sim_i_j = sim_matrix[i, j]

                numerator = tf.exp(sim_i_j / self.temperature)
                one_for_not_i = tf.tensor_scatter_nd_update(tf.ones(2 * self.batch_size), [[i]], [0.])
                denominator = tf.reduce_sum(one_for_not_i * tf.exp(sim_matrix[i, :] / self.temperature))

                loss_ij = -tf.math.log(numerator / denominator)
                return tf.squeeze(loss_ij, axis=0)

            N = self.batch_size
            loss = 0.
            for k in range(0, N):
                loss += get_loss_ij(k, k + N) + get_loss_ij(k + N, k)

            loss = 1 / (2 * N) * loss

        elif how == 'fast':
            sim_ij = tf.linalg.diag_part(sim_matrix, k=self.batch_size)
            sim_ji = tf.linalg.diag_part(sim_matrix, k=-self.batch_size)
            positives = tf.concat([sim_ij, sim_ji], axis=0)

            nominator = tf.math.exp(positives / self.temperature)
            denominator = self.negative_mask * tf.math.exp(sim_matrix / self.temperature)

            loss_partial = -tf.math.log(nominator / tf.reduce_sum(denominator, axis=1))
            loss = tf.reduce_sum(loss_partial) / (2 * self.batch_size)

        return loss

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker
        ]

    @tf.function
    def call(self, inputs, training=None, mask=None):
        outputs = self.encoder(inputs)
        return outputs

    @tf.function
    def train_step(self, data, how: str = 'naive'):
        anchor_inputs, target_inputs = data
        with tf.GradientTape() as tape:
            anchor_embeddings = self(anchor_inputs)
            target_embeddings = self(target_inputs)
            contrastive_loss = self.get_contrastive_loss(anchor_embeddings, target_embeddings)

        gradients = tape.gradient(contrastive_loss, self.trainable_variables)
        self.contrastive_optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.contrastive_loss_tracker.update_state(contrastive_loss)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        anchor_inputs, target_inputs = data

        anchor_embeddings = self(anchor_inputs)
        target_embeddings = self(target_inputs)
        contrastive_loss = self.get_contrastive_loss(anchor_embeddings, target_embeddings)

        self.contrastive_loss_tracker.update_state(contrastive_loss)
        return {m.name: m.result() for m in self.metrics}
