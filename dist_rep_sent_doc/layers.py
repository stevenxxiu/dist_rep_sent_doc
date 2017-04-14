import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops

__all__ = ['HierarchicalSoftmaxLayer']


# noinspection PyProtectedMember,PyAttributeOutsideInit
class HierarchicalSoftmaxLayer(base._Layer):
    def __init__(
        self, tree, node_id_to_index, W_initializer=init_ops.zeros_initializer(),
        name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.tree = tree
        self.node_id_to_index = node_id_to_index
        self.W_initializer = W_initializer

        # training
        node_to_path = self._get_node_to_path(tree)
        self.shape = (len(node_to_path), max(len(path) for path in node_to_path.values()))
        nodes, signs, masks = \
            np.zeros(self.shape, dtype=np.int32), \
            np.zeros(self.shape, dtype=np.float32), \
            np.zeros(self.shape, dtype=np.bool)
        for i in range(len(node_to_path)):
            for j, (node, sign) in enumerate(node_to_path[i]):
                nodes[i, j] = node
                signs[i, j] = sign
                masks[i, j] = True
        indices = [
            [node, (0 if sign == 1 else len(node_id_to_index) - 1) + node_]
            for node, path in node_to_path.items() for node_, sign in path
        ]
        self.nodes, self.signs, self.masks = tf.constant(nodes), tf.constant(signs), tf.constant(masks)

        # output
        self.output_index = tf.sparse_reorder(tf.SparseTensor(
            indices=indices, values=tf.ones([len(indices)]),
            dense_shape=[len(node_to_path), 2 * (len(node_to_path) - 1)]
        ))

    def _get_node_to_path(self, tree):
        '''
        :return: A dictionary mapping leaves in the tree to it's path.
        '''
        if tree.left:
            return {
                node: [(tree.id_, 1 if i == 0 else -1)] + path
                for i, child in enumerate([tree.left, tree.right])
                for node, path in self._get_node_to_path(child).items()
            }
        else:
            return {self.node_id_to_index[tree.id_]: []}

    def _compute_output_shape(self, input_shape):
        return None

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape[0] if isinstance(input_shape, list) else input_shape)
        self.W = vs.get_variable(
            'W', shape=[len(self.node_id_to_index) - 1, input_shape[-1].value],
            initializer=self.W_initializer
        )

    def call(self, inputs, training=False):
        if training:
            input_, target = inputs
            masks = tf.reshape(tf.gather(self.masks, target), [-1])
            nodes = tf.boolean_mask(tf.reshape(tf.gather(self.nodes, target), [-1]), masks)
            signs = tf.boolean_mask(tf.reshape(tf.gather(self.signs, target), [-1]), masks)
            indices = tf.boolean_mask(tf.reshape(
                tf.tile(tf.reshape(tf.range(tf.shape(input_)[0]), [-1, 1]), [1, self.shape[1]]), [-1]
            ), masks)
            return tf.reduce_sum(-tf.nn.softplus(
                -signs * tf.reduce_sum(tf.gather(self.W, nodes) * tf.gather(input_, indices), 1)
            ))
        else:
            node_outputs = tf.matmul(inputs, tf.transpose(self.W))
            return tf.transpose(tf.sparse_tensor_dense_matmul(self.output_index, tf.transpose(
                -tf.nn.softplus(-tf.concat([node_outputs, -node_outputs], 1))
            )))
