import numpy as np
import lasagne
import theano.tensor as T
from lasagne import init

__all__ = ['HierarchicalSoftmaxLayer']


class HierarchicalSoftmaxLayer(lasagne.layers.Layer):
    def __init__(
        self, incoming, tree, node_id_to_index, W=init.GlorotUniform(), b=init.Constant(0.), **kwargs
    ):
        super().__init__(incoming, **kwargs)
        self.tree = tree
        self.node_id_to_index = node_id_to_index
        self.node_to_path = self._get_node_to_path(tree)
        self.W = self.add_param(W, (len(node_id_to_index), incoming.output_shape[-1]), name='W')
        self.b = self.add_param(b, (len(node_id_to_index),), name='b')

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

    def get_output_shape_for(self, input_shape):
        # depends on wehther we have a target
        return None

    def get_output_for(self, input_, hs_nodes=None, hs_signs=None, hs_indexes=None, **kwargs):
        # sparse matrices are not implemented on the gpu yet
        if hs_nodes is None:
            return T.log(T.nnet.sigmoid(T.stack([
                -(T.dot(input_, self.W.T) + self.b),
                T.dot(input_, self.W.T) + self.b
            ])))
        else:
            return T.log(T.nnet.sigmoid(hs_signs * (
                (self.W[hs_nodes] * input_[hs_indexes]).sum(1) +
                self.b[hs_nodes]
            ))).sum()

    def get_hs_inputs(self, target):
        hs_nodes, hs_signs, hs_indexes = [], [], []
        for i, target_ in enumerate(target):
            for node, sign in self.node_to_path[target_]:
                hs_nodes.append(node)
                hs_signs.append(sign)
                hs_indexes.append(i)
        return hs_nodes, hs_signs, hs_indexes

    def get_hs_outputs(self, output):
        res = np.zeros((len(output), len(self.node_id_to_index)))
        for i, output_ in enumerate(output):
            for j in range(len(self.node_id_to_index)):
                for node, sign in self.node_to_path[j]:
                    res[i][j] += output[0 if sign == -1 else 1][i][node]
        return res
