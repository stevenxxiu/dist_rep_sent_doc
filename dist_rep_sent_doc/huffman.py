import heapq

__all__ = ['HuffmanNode', 'build_huffman']


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
    while len(queue) > 1:
        children = [heapq.heappop(queue), heapq.heappop(queue)]
        heapq.heappush(queue, HuffmanNode(len(queue), children[0].freq + children[1].freq, *children))
    return queue[0]
