import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops

__all__ = ['HierarchicalSoftmaxLayer']


# noinspection PyProtectedMember,PyAttributeOutsideInit
class HierarchicalSoftmaxLayer(base._Layer):
    def __init__(self, tree, W_initializer=init_ops.zeros_initializer(), name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.tree = tree
        self.W_initializer = W_initializer
        nodes, signs, masks = tree
        self.nodes, self.signs, self.masks = tf.constant(nodes), tf.constant(signs), tf.constant(masks)
        self.output_index = tf.SparseTensor(indices=np.array([
            np.repeat(np.arange(len(nodes)), nodes.shape[1])[np.ravel(masks)],
            (np.ravel(nodes) + (np.ravel(signs) == -1) * (len(nodes) - 1))[np.ravel(masks)],
        ]).T, values=tf.ones([np.sum(masks)]), dense_shape=[len(self.tree[0]), 2 * (len(self.tree[0]) - 1)])

    def _compute_output_shape(self, input_shape):
        return None

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape[0] if isinstance(input_shape, list) else input_shape)
        self.W = vs.get_variable(
            'W', shape=[len(self.tree[0]) - 1, input_shape[-1].value],
            initializer=self.W_initializer
        )

    def call(self, inputs, training=False):
        if training:
            input_, target = inputs
            masks = tf.reshape(tf.gather(self.masks, target), [-1])
            nodes = tf.boolean_mask(tf.reshape(tf.gather(self.nodes, target), [-1]), masks)
            signs = tf.boolean_mask(tf.reshape(tf.gather(self.signs, target), [-1]), masks)
            indices = tf.boolean_mask(tf.reshape(
                tf.tile(tf.reshape(tf.range(tf.shape(input_)[0]), [-1, 1]), [1, self.tree[0].shape[1]]), [-1]
            ), masks)
            return tf.reduce_sum(-tf.nn.softplus(
                -signs * tf.reduce_sum(tf.nn.embedding_lookup(self.W, nodes) * tf.gather(input_, indices), 1)
            ))
        else:
            node_outputs = tf.matmul(inputs, tf.transpose(self.W))
            return tf.transpose(tf.sparse_tensor_dense_matmul(self.output_index, tf.transpose(
                -tf.nn.softplus(-tf.concat([node_outputs, -node_outputs], 1))
            )))
