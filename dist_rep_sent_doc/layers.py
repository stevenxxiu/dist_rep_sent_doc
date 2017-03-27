import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base, utils
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops


# noinspection PyProtectedMember,PyAttributeOutsideInit
class HierarchicalSoftmaxLayer(base._Layer):
    def __init__(
        self, tree, node_id_to_index,
        W_initializer=init_ops.glorot_uniform_initializer(), b_initializer=init_ops.zeros_initializer(),
        trainable=True, name=None, **kwargs
    ):
        super().__init__(trainable=trainable, name=name, **kwargs)
        self.tree = tree
        self.node_id_to_index = node_id_to_index
        self.W_initializer = W_initializer
        self.b_initializer = b_initializer

    def _get_node_to_path(self, tree):
        '''
        :return: A dictionary mapping leaves in the tree to it's path.
        '''
        if tree.left:
            return {
                node: [(tree.id_, -1 if i == 0 else 1)] + path
                for i, child in enumerate([tree.left, tree.right])
                for node, path in self._get_node_to_path(child).items()
            }
        else:
            return {self.node_id_to_index[tree.id_]: []}

    def build(self, input_shape):
        node_to_path = self._get_node_to_path(self.tree)
        emb_dim = input_shape[0][-1].value
        self.W = vs.get_variable(
            'W', shape=[len(self.node_id_to_index), emb_dim],
            initializer=self.W_initializer, dtype=self.dtype, trainable=self.trainable
        )
        self.b = vs.get_variable(
            'b', shape=[len(self.node_id_to_index)],
            initializer=self.b_initializer, dtype=self.dtype, trainable=self.trainable
        )
        max_path = max(len(path) for path in node_to_path.values())
        self.path_signs = np.zeros((len(self.node_id_to_index), max_path))
        indices, data_indices = [], []
        for i in range(len(self.node_id_to_index)):
            for j, (node, sign) in enumerate(node_to_path[i]):
                indices.extend([i, j, k] for k in range(emb_dim))
                data_indices.append(node)
                self.path_signs[i][j] = sign
        self.path_W = tf.SparseTensor(
            indices=indices, values=tf.reshape(tf.gather(self.W, data_indices), [-1]),
            dense_shape=(len(self.node_id_to_index), max_path, emb_dim)
        )

    def call(self, inputs, training=False):
        # # XXX try out sparse matrices to see if there's a speed up
        # return utils.smart_cond(
        #     training,
        #     lambda: tf.reduce_sum(tf.log(tf.nn.sigmoid(self.hs_signs * (
        #         tf.reduce_sum(tf.gather(self.W, self.hs_nodes) * tf.gather(inputs, self.hs_indexes), 1) +
        #         tf.gather(self.b, self.hs_nodes)
        #     )))) / tf.cast(tf.shape(inputs)[0], tf.float32),
        #     lambda: tf.log(tf.nn.sigmoid(tf.stack([
        #         -(tf.matmul(inputs, tf.transpose(self.W)) + self.b),
        #         tf.matmul(inputs, tf.transpose(self.W)) + self.b
        #     ])))
        # )
        X, y = inputs
        return tf.gather(self.path_W, y)

    # def get_hs_outputs(self, output):
    #     res = np.zeros((len(output), len(self.node_id_to_index)))
    #     for i, output_ in enumerate(output):
    #         for j in range(len(self.node_id_to_index)):
    #             for node, sign in self.node_to_path[j]:
    #                 res[i][j] += output[0 if sign == -1 else 1][i][node]
    #     return res
