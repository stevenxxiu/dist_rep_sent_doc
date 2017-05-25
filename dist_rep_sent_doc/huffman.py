import heapq

import numpy as np

__all__ = ['HuffmanNode', 'build_huffman', 'tree_to_arrays']


class HuffmanNode:
    def __init__(self, id_, freq, left=None, right=None):
        self.id_ = id_
        self.freq = freq
        self.left = left
        self.right = right

    def __eq__(self, other):
        return self.freq == other.freq and self.id_ == other.id_ and \
               self.left == other.left and self.right == other.right

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman(word_to_freq):
    queue = [HuffmanNode(word, freq) for word, freq in word_to_freq.items()]
    heapq.heapify(queue)
    for i in range(len(word_to_freq) - 1):
        children = [heapq.heappop(queue), heapq.heappop(queue)]
        heapq.heappush(queue, HuffmanNode(i, children[0].freq + children[1].freq, *children))
    return queue[0]


def tree_to_arrays(root, node_id_to_index, size):
    nodes, signs, masks = [[None] * size for _ in range(3)]
    stack, visited, max_depth = [root], set(), 0
    while stack:
        node = stack[-1]
        if node.left:
            if node.left.id_ not in visited:
                stack.append(node.left)
                continue
            if node.right.id_ not in visited:
                stack.append(node.right)
                continue
        else:
            id_ = node_id_to_index[stack[-1].id_]
            nodes[id_] = [node.id_ for node in stack[:-1]]
            signs[id_] = np.float32([1 if stack[i + 1] is stack[i].left else -1 for i in range(len(stack) - 1)])
            masks[id_] = [True] * (len(stack) - 1)
            max_depth = max(max_depth, len(stack) - 1)
        stack.pop()
        visited.add(node.id_)
    return [np.vstack([np.pad(v, (0, max_depth - len(v)), 'constant') for v in X]) for X in (nodes, signs, masks)]
