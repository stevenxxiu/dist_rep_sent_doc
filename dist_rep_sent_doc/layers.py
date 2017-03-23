import lasagne
import numpy as np
import theano.tensor as T
from lasagne import init

__all__ = ['HierarchicalSoftmaxLayer']


class HierarchicalSoftmaxLayer(lasagne.layers.Layer):
    def __init__(
        self, incoming, tree, word_to_index, W=init.GlorotUniform(), b=init.Constant(0.), **kwargs
    ):
        super().__init__(incoming, **kwargs)
        mask, path, child = self._tree_to_mat(tree, word_to_index)
        self.mask = self.add_param(mask, mask.shape, name='mask', trainable=False)
        self.path = self.add_param(path, path.shape, name='path', trainable=False)
        self.child = self.add_param(child, child.shape, name='child', trainable=False)
        self.W = self.add_param(W, (len(word_to_index), incoming.output_shape[-1]), name='W')
        self.b = self.add_param(b, (len(word_to_index),), name='b')

    @classmethod
    def _tree_to_paths(cls, tree):
        '''
        :return: A dictionary mapping words in the tree to it's path.
        '''
        if tree.left:
            return {
                word: [(tree.id_, -1 if i == 0 else 1)] + path
                for i, child in enumerate([tree.left, tree.right])
                for word, path in cls._tree_to_paths(child).items()
            }
        else:
            return {tree.id_: []}

    @classmethod
    def _tree_to_mat(cls, tree, word_to_index):
        '''
        Converts a huffman tree to:
        - A mask matrix where each row is a binary mask for the other matrices.
        - A path matrix where each row contains the path from the root.
        - A child matrix where -1 and 1 represent the left and right child respectively.
        '''
        word_to_path = cls._tree_to_paths(tree)
        d = max(len(path) for path in word_to_path.values())
        mask_mat = np.zeros((len(word_to_index), d))
        path_mat = np.zeros((len(word_to_index), d), dtype=int)
        child_mat = np.zeros((len(word_to_index), d))
        for word, path in word_to_path.items():
            mask_mat[word_to_index[word], :len(path)] = 1
            path_mat[word_to_index[word], :len(path)] = [path_item[0] for path_item in path]
            child_mat[word_to_index[word], :len(path)] = [path_item[1] for path_item in path]
        return mask_mat, path_mat, child_mat

    def get_output_shape_for(self, input_shape):
        # depends on wehther we have a target
        return None

    def get_output_for(self, input_, hs_target=None, **kwargs):
        # sparse matrices are not implemented on the gpu yet
        if hs_target is None:
            return T.sum(self.mask.dimshuffle(('x', 0, 1)) * T.log(T.nnet.sigmoid(
                self.child.dimshuffle(('x', 0, 1)) * (
                    T.dot(self.W[self.path], input_.T).dimshuffle((2, 0, 1)) +
                    self.b[self.path].dimshuffle(('x', 0, 1))
                )
            )), axis=-1)
        else:
            return T.sum(self.mask[hs_target] * T.log(T.nnet.sigmoid(
                self.child[hs_target] * (
                    T.batched_dot(self.W[self.path[hs_target]], input_) +
                    self.b[self.path[hs_target]]
                )
            )), axis=-1)
