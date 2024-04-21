import numpy as np
import pandas as pd
import tensorflow as tf
#
# a = tf.constant([[0,1], [1,1], [1,1]], dtype=tf.float32)
# b1 = tf.constant([[1,0], [1,1], [0,0]], dtype=tf.float32)
# b2 = tf.constant([[0,1], [-1,-1], [2,2]], dtype=tf.float32)
#
# b = tf.stack((b1,b2), axis=0)
# print(b)
# print(b.shape)
#
# sim = tf.keras.losses.CosineSimilarity(
#     axis=-1, reduction=tf.keras.losses.Reduction.NONE)
# cos_sim = sim(tf.broadcast_to(a, b.shape), b)
# print(cos_sim)

batch_size = 4
negative_mask = np.ones((2 * batch_size, 2 * batch_size))
negative_mask[:batch_size, :batch_size] = 0
negative_mask[-batch_size:, -batch_size:] = 0
negative_mask = tf.constant(negative_mask, dtype=tf.float32)
print(negative_mask)