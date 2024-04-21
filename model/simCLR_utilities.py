import keras.metrics
import numpy as np
from config_parser import Parser

import tensorflow as tf
from keras.layers import (Input, LSTM, Conv2D, Conv1D, Dense,
                          Dropout, GlobalMaxPool1D, Activation, ReLU,
                          MaxPooling1D, ZeroPadding1D, Flatten, LayerNormalization, Layer,
                          BatchNormalization)
from keras.models import Model
from tensorflow.math import l2_normalize
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras import initializers
import keras.backend as K
from tcn import TCN


def conv1D_Block(inputs, n: int, use_dropout: bool = False, kernel_size: int = 3, filters: int = 64, use_pooling: bool = True):
    with K.name_scope('conv1D_Block_' + str(n)):
        padding = ZeroPadding1D(padding=1, name='ZeroPadding_' + str(n))
        conv1D = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='valid',
            kernel_initializer=initializers.he_uniform(),
            name='Conv1D_' + str(n)
        )
        relu = ReLU(name='ReLU_' + str(n))
        dropout = Dropout(rate=0.1, name='Dropout_' + str(n))
        pooling = MaxPooling1D(2, strides=2, name='MaxPooling_' + str(n))

        x = padding(inputs)
        x = conv1D(x)
        x = relu(x)
        if use_dropout:
            x = dropout(x)
        if use_pooling:
            x = pooling(x)

    return x


class channel_attention(tf.keras.layers.Layer):
    def __init__(self, n_filters, kernel_size, dilation_rate):
        super(channel_attention, self).__init__()
        self.conv_1 = Conv2D(n_filters, kernel_size=kernel_size,
                             padding='same', activation='relu', dilation_rate=dilation_rate)
        self.conv_f = Conv2D(1, kernel_size=1, padding='same')
        self.ln = LayerNormalization()

    def call(self, x):
        x = self.ln(x)
        x1 = tf.expand_dims(x, axis=3)
        x1 = self.conv_1(x1)
        x1 = self.conv_f(x1)
        x1 = tf.keras.activations.softmax(x1, axis=2)
        x1 = tf.keras.layers.Reshape(x.shape[-2:])(x1)

        return tf.math.multiply(x, x1), x1


class positional_encoding(Layer):

    def __init__(self, n_timesteps, n_features):
        super(positional_encoding, self).__init__()
        self.pos_encoding = self.positional_encoding(n_timesteps, n_features)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, n_timesteps, n_features):
        angle_rads = self.get_angles(
            position=tf.range(n_timesteps, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(n_features, dtype=tf.float32)[tf.newaxis, :],
            d_model=n_features)

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def scaled_dot_product_attention(q, k, v):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHead_attention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHead_attention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, use_bias=False)
        self.wk = tf.keras.layers.Dense(d_model, use_bias=True)
        self.wv = tf.keras.layers.Dense(d_model, use_bias=True)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


class transformer_layer(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(transformer_layer, self).__init__()

        self.mha = MultiHead_attention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        attn_output, _ = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(out1 + ffn_output)

        return out2


class global_temporal_attention(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False):
        super(global_temporal_attention, self).__init__()

        self.supports_masking = True
        self.return_attention = return_attention
        self.init = tf.keras.initializers.get('glorot_uniform')

        self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
        self.u_regularizer = tf.keras.regularizers.get(u_regularizer)
        self.b_regularizer = tf.keras.regularizers.get(b_regularizer)

        self.W_constraint = tf.keras.constraints.get(W_constraint)
        self.u_constraint = tf.keras.constraints.get(u_constraint)
        self.b_constraint = tf.keras.constraints.get(b_constraint)

        self.bias = bias

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x):
        uit = tf.tensordot(x, self.W, axes=1)

        if self.bias:
            uit += self.b

        uit = tf.keras.activations.tanh(uit)
        ait = tf.tensordot(uit, self.u, axes=1)

        a = tf.math.exp(ait)

        a /= tf.cast(tf.keras.backend.sum(a, axis=1, keepdims=True) + tf.keras.backend.epsilon(),
                     tf.keras.backend.floatx())

        a = tf.keras.backend.expand_dims(a)
        weighted_input = x * a
        result = tf.keras.backend.sum(weighted_input, axis=1)

        if self.return_attention:
            return result, a
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return tf.TensorShape([input_shape[0].value, input_shape[-1].value],
                                  [input_shape[0].value, input_shape[1].value])
        else:
            return tf.TensorShape([input_shape[0].value, input_shape[-1].value])


class simCLR(Model):
    def __init__(self, inputs_shape: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.contrastive_loss_tracker = None
        self.contrastive_optimizer = None

        config = Parser()
        config.get_args()
        self.conf = config

        self.inputs_shape = inputs_shape
        self.L = 100
        self.win = inputs_shape[1]

        if self.conf.encoder == 'TCN':
            self.anchor_encoder = self.get_TCN_encoder(sensor='anchor')
            self.target_encoder = self.get_TCN_encoder(sensor='target')
        elif self.conf.encoder == 'CNN':
            self.anchor_encoder = self.get_CNN_encoder(sensor='anchor')
            self.target_encoder = self.get_CNN_encoder(sensor='target')
        elif self.conf.encoder == 'Attention':
            self.anchor_encoder = self.get_attention_encoder(sensor='anchor')
            self.target_encoder = self.get_attention_encoder(sensor='target')

        if self.conf.attach_head:
            self.anchor_encoder = self.attach_head(hidden_layers = [self.L], sensor='anchor')
            self.target_encoder = self.attach_head(hidden_layers = [self.L], sensor='target')

        self.batch_size = self.conf.batch_size
        self.temperature = self.conf.clr_temp

        if self.conf.neg_pos == 'other':
            self.negative_mask = np.ones((2 * self.batch_size, 2 * self.batch_size))
            self.negative_mask[:self.batch_size, :self.batch_size] = 0
            self.negative_mask[-self.batch_size:, -self.batch_size:] = 0
            self.negative_mask = tf.constant(self.negative_mask, dtype=tf.float32)
        elif self.conf.neg_pos == 'all':
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
        self.anchor_encoder.summary()

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

    def get_attention_encoder(self, sensor: str, name: str = 'attention_encoder'):
        inputs = Input(shape=self.inputs_shape[1:])

        x = Conv1D(filters=32, kernel_size=3, strides=1, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv1D(filters=128, kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = positional_encoding(n_timesteps=self.inputs_shape[1], n_features=128)(x)
        x = Dropout(rate=0.1)(x)

        for _ in range(6):
            x = transformer_layer(d_model=128, num_heads=8, dff=256, rate=0.1)(x)

        x = Dropout(rate=0.1)(x)
        x = Flatten()(x)

        return Model(inputs, x, name = sensor + '_' + name)

        # ci, c_attention = channel_attention(n_filters=128, kernel_size=3, dilation_rate=2)(inputs)
        #
        # x = Conv1D(filters=128, kernel_size=1, activation='relu')(ci)
        #
        # x *= tf.math.sqrt(tf.cast(128, tf.float32))
        # x = positional_encoding(n_timesteps=self.inputs_shape[1], n_features=128)(x)
        # x = Dropout(rate=0.2)(x)
        #
        # x = transformer_layer(d_model=128, num_heads=4, rate=0.2, dff=512)(x)
        # x = transformer_layer(d_model=128, num_heads=4, rate=0.2, dff=512)(x)
        #
        # x = global_temporal_attention()(x)
        #
        # return Model(inputs, x, name = sensor + '_' + name)

    def get_CNN_encoder(self, sensor: str, name: str = 'CNN_encoder') -> Model:
        inputs = Input(shape=self.inputs_shape[1:])
        x = inputs

        filters = [64, 64, 64]
        kernels = [3, 3, 3]
        pooling = [True, True, True]
        dropout = [False, False, False]

        for i in range(len(filters)):
            x = conv1D_Block(x, i+1, filters=filters[i], kernel_size=kernels[i],
                             use_pooling=pooling[i], use_dropout=dropout[i])

        x = Flatten()(x)

        return Model(inputs, x, name= sensor + '_' + name)

    def get_TCN_encoder(self, sensor: str, name: str = 'TCN_encoder') -> Model:
        inputs = Input(shape=self.inputs_shape[1:], name = 'input')
        x = inputs

        tcn_layer = TCN(10, 4, 10, [1, 4, 8, 16], padding='causal', use_skip_connections=True,
                dropout_rate=0.1, return_sequences=True, activation='relu', kernel_initializer='random_normal')
        x = tcn_layer(x)
        x = Flatten()(x)

        return Model(inputs, x, name=sensor + '_' + name)

    def attach_head(self, sensor: str, hidden_layers = [], name: str = 'projection_encoder') -> Model:
        if sensor == 'anchor':
            inputs = self.anchor_encoder.input
            x = self.anchor_encoder.output
        elif sensor == 'target':
            inputs = self.target_encoder.input
            x = self.target_encoder.output

        N = len(hidden_layers)

        for i, hidden in enumerate(hidden_layers):
            x = Dense(hidden)(x)
            if i < N - 1:
                x = Activation('relu')(x)

        simclr_model = Model(inputs, x, name = sensor + '_' + name)

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
        outputs = self.anchor_encoder(inputs)
        return outputs

    @tf.function
    def train_step(self, data, how: str = 'naive'):
        anchor_inputs, target_inputs = data
        with tf.GradientTape() as tape:
            anchor_embeddings = self.anchor_encoder(anchor_inputs)
            target_embeddings = self.target_encoder(target_inputs)
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
