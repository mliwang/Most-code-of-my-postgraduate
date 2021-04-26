"""
Classes describing datasets of user-item interactions. Instances of these
are returned by dataset-fetching and dataset-processing functions.
"""

import numpy as np

import scipy.sparse as sp


class Interactions(object):
    """
    Interactions object. Contains (at a minimum) pair of user-item
    interactions. This is designed only for implicit feedback scenarios.
    交互对象。包含(至少)一对用户-项目交互。这只是为隐式反馈场景设计的。

    Parameters
    ----------

    file_path: file contains (user,item,rating) triplets
    user_map: dict of user mapping
    item_map: dict of item mapping
    """

    def __init__(self, file_path,
                 user_map=None,
                 item_map=None):

        if not user_map and not item_map:
            user_map = dict()
            item_map = dict()

            num_user = 0
            num_item = 0
        else:
            num_user = len(user_map)
            num_item = len(item_map)

        user_ids = list()
        item_ids = list()
        # 逐行读取数据，只读取clientIP和URL
        with open(file_path, 'r') as fin:
            for line in fin:
                #u, i, _ = line.strip().split()
                u, i, _, _ = line.strip().split()
                user_ids.append(u)
                item_ids.append(i)

        # 对clientIP和URL的词典
        for u in user_ids:
            if u not in user_map:
                user_map[u] = num_user
                num_user += 1
        for i in item_ids:
            if i not in item_map:
                item_map[i] = num_item
                num_item += 1

        #将原始IP和URL转化为ID
        user_ids = np.array([user_map[u] for u in user_ids])
        item_ids = np.array([item_map[i] for i in item_ids])

        # print(item_ids)
        # print(num_item)
        '''user_ids应该是所有用户的编号
           item_ids应该是所有商品的编号
           print(user_ids)
           print(user_map)
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4]
{'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
           
           
        '''
        #词典长度
        self.num_users = num_user
        self.num_items = num_item

        #数据存储列
        self.user_ids = user_ids
        self.item_ids = item_ids

        #词典
        self.user_map = user_map
        self.item_map = item_map

        self.sequences = None
        self.test_sequences = None

    def __len__(self):

        return len(self.user_ids)

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        COO使用3个数组进行存储：values,rows, andcolumn。
        其中
        数组values: 实数或复数数据，包括矩阵中的非零元素，顺序任意。
        数组rows: 数据所处的行。
        数组columns: 数据所处的列。
        """
        #coo（Coordinate,坐标）矩阵：包含列号、行号、值三个list，把矩阵中不为0的数据的行号和列号存储下来，转化为其他稀疏矩阵存储形式（如CSR）
        row = self.user_ids
        col = self.item_ids
        data = np.ones(len(self))

        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix. 压缩稀疏行格式(CSR)

        """
        # CSR（Compressed Sparse Row, 压缩稀疏行）矩阵：包含行偏移、列号、值三个list，
        #row offset的数值个数是#row + 1, 表示某行第一个元素在values中的位置，如5是第三行第一个元素，它在values中的index是4。
        return self.tocoo().tocsr()

    def to_sequence(self, sequence_length=5, target_length=1):
        """
        Transform to sequence form.

        Valid subsequences of users' interactions are returned. For
        example, if a user interacted with items [1, 2, 3, 4, 5, 6, 7, 8, 9], the
        returned interactions matrix at sequence length 5 and target length 3
        will be be given by:

        sequences:

           [[1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]]

        targets:

           [[6, 7],
            [7, 8],
            [8, 9]]

        sequence for test (the last 'sequence_length' items of each user's sequence):

        [[5, 6, 7, 8, 9]]  test_sequences

        Parameters
        ----------

        sequence_length: int
            Sequence length. Subsequences shorter than this
            will be left-padded with zeros.
        target_length: int
            Sequence target length.
        """

        # 为了对齐序列，而将URL ID从1开始
        for k, v in self.item_map.items():
            self.item_map[k] = v + 1
        self.item_ids = self.item_ids + 1
        self.num_items += 1
        # print( self.item_ids)
        # print( self.num_items)

        #最长序列的长度
        max_sequence_length = sequence_length + target_length


        # 使用用户IP的ID 对原始样本进行排序
        #numpy.lexsort() 用于对多个序列进行排序。把它想象成对电子表格进行排序，每一列代表一个序列，排序时优先照顾靠后的列。
        sort_indices = np.lexsort((self.user_ids,))

        user_ids = self.user_ids[sort_indices]
        item_ids = self.item_ids[sort_indices]

        #unique（）保留数组中不同的值，返回两个参数（元素，元素出现的起始位置）
        user_ids, indices, counts = np.unique(user_ids,
                                              return_index=True,
                                              return_counts=True)


        #num_subsequences计算的是序列总数
        num_subsequences = sum([c - max_sequence_length + 1 if c >= max_sequence_length else 1 for c in counts])
        # print(self.num_users)
        '''
         为什么下面的矩阵会以num_subsequences为矩阵的形状
         比如用户1的历史行为是[0，1，2，3，4，5，6，7，8，9] 
         那么根据计算c=10-8+1=3 会产生3个序列 
         [0,1,2,3,4]->[5,6,7]
         [1,2,3,4,5]->[6,7,8]
         [2,3,4,5,6]->[7,8,9]
         num_subsequences计算的是序列总数
        '''

        #特征序列
        sequences = np.zeros((num_subsequences, sequence_length),
                             dtype=np.int64)
        #目标序列
        sequences_targets = np.zeros((num_subsequences, target_length),
                                     dtype=np.int64)
        #用户序列
        sequence_users = np.empty(num_subsequences,
                                  dtype=np.int64)

        #测试特征序列
        test_sequences = np.zeros((self.num_users, sequence_length),
                                  dtype=np.int64)
        # 测试用户序列  test_sequences [5,5]
        test_users = np.empty(self.num_users,
                              dtype=np.int64)

        _uid = None
        #使用移动窗口：一维序列转二维
        for i, (uid,item_seq) in enumerate(_generate_sequences(user_ids,item_ids,indices,max_sequence_length)):
            # print(uid,item_seq) test_sequences代表的是用户最后的5个item
            #如果当前用户ID不等于上一个用户ID，则输出序列
            if uid != _uid:
                #print ("item_seq:", item_seq)
                test_sequences[uid][:] = item_seq[-sequence_length:]
                test_users[uid] = uid
                _uid = uid
                # print(test_sequences)
            sequences_targets[i][:] = item_seq[-target_length:]
            sequences[i][:] = item_seq[:sequence_length]
            sequence_users[i] = uid

        #构造训练样例和测试样例
        self.sequences = SequenceInteractions(sequence_users, sequences, sequences_targets)
        self.test_sequences = SequenceInteractions(test_users, test_sequences)



class SequenceInteractions(object):
    """
    Interactions encoded as a sequence matrix. 转化为矩阵

    Parameters
    ----------
    user_ids: np.array
        sequence users
    sequences: np.array
        The interactions sequence matrix, as produced by
        :func:`~Interactions.to_sequence`
    targets: np.array
        sequence targets
    """

    def __init__(self,user_ids,sequences,targets=None):
        self.user_ids = user_ids
        self.sequences = sequences
        self.targets = targets

        self.L = sequences.shape[1]
        self.T = None
        if np.any(targets):
            self.T = targets.shape[1]
        # print(self.sequences)
        # print(self.targets)


def _sliding_window(tensor, window_size, step_size=1):
    #如果两个index 的间隔大于window_size
    if len(tensor) - window_size >= 0:
        #产生两个index间隔的递减序列，直到中间某个位置等于windows_size
        for i in range(len(tensor), 0, -step_size):
            if i - window_size >= 0:
                yield tensor[i - window_size:i]
            else:
                break
    else:
        num_paddings = window_size - len(tensor)
        # Pad sequence with 0s if it is shorter than windows size.
        yield np.pad(tensor, (num_paddings, 0), 'constant')


def _generate_sequences(user_ids, item_ids,indices,max_sequence_length):
    for i in range(len(indices)):
        start_idx = indices[i]
        if i >= len(indices) - 1:
            stop_idx = None
        else:
            stop_idx = indices[i + 1]
        #每两个index之间的位置进行产生
        for seq in _sliding_window(item_ids[start_idx:stop_idx],max_sequence_length):
            #产生第i个user_id和一个URL的递减序列
            yield (user_ids[i], seq)
