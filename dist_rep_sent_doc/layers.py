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
        self.node_to_path = self._get_node_to_path(tree)
        self.W_initializer = W_initializer
        self.b_initializer = b_initializer
        self.hs_nodes = tf.placeholder(tf.int32, [None])
        self.hs_signs = tf.placeholder(tf.float32, [None])
        self.hs_indexes = tf.placeholder(tf.int32, [None])

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
        input_shape = tensor_shape.TensorShape(input_shape)
        self.W = vs.get_variable(
            'W', shape=[len(self.node_id_to_index), input_shape[-1].value],
            initializer=self.W_initializer, dtype=self.dtype, trainable=self.trainable
        )
        self.b = vs.get_variable(
            'b', shape=[len(self.node_id_to_index)],
            initializer=self.b_initializer, dtype=self.dtype, trainable=self.trainable
        )

    def call(self, inputs, training=False):
        return utils.smart_cond(
            training,
            lambda: tf.reduce_sum(tf.log(tf.nn.sigmoid(self.hs_signs * (
                tf.reduce_sum(tf.gather(self.W, self.hs_nodes) * tf.gather(inputs, self.hs_indexes), 1) +
                tf.gather(self.b, self.hs_nodes)
            )))) / tf.cast(tf.shape(inputs)[0], tf.float32),
            lambda: tf.log(tf.nn.sigmoid(tf.stack([
                -(tf.matmul(inputs, tf.transpose(self.W)) + self.b),
                tf.matmul(inputs, tf.transpose(self.W)) + self.b
            ])))
        )

    def get_hs_inputs(self, target):
        hs_nodes, hs_signs, hs_indexes = [], [], []
        for i, target_ in enumerate(target):
            for node, sign in self.node_to_path[target_]:
                hs_nodes.append(node)
                hs_signs.append(sign)
                hs_indexes.append(i)
        return {self.hs_nodes: hs_nodes, self.hs_signs: hs_signs, self.hs_indexes: hs_indexes}

    def get_hs_outputs(self, output):
        res = np.zeros((len(output), len(self.node_id_to_index)))
        for i, output_ in enumerate(output):
            for j in range(len(self.node_id_to_index)):
                for node, sign in self.node_to_path[j]:
                    res[i][j] += output[0 if sign == -1 else 1][i][node]
        return res
