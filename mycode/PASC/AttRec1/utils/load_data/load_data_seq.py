import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


class DataSet():
    class SeqData():

        def __init__(self, user_ids, sequences, targets=None):
            self.user_ids = user_ids
            self.sequences = sequences
            self.targets = targets
            self.L = sequences.shape[1]
            self.T = None

            if np.any(targets):
                self.T = targets.shape[1]

    def __init__(self, path="../../Data/ml1m/seq/train.txt", header=['user', 'item', 'rating'], sep=" ", seq_len=1,
                 target_len=1, isTrain=False, user_map=None,
                 item_map=None, num_users=None, num_items=None):
        self.path = path
        self.header = header
        self.sep = sep
        self.seq_len = seq_len
        self.target_len = target_len
        self.isTrain = isTrain

        if not user_map and not item_map:
            user_map = dict()
            item_map = dict()

            self.num_user = 0
            self.num_item = 0
        else:
            self.num_user = len(user_map)
            self.num_item = len(item_map)

        # TODO: 1. remove cold start user with less than 5 items;
        #移除交互物品少于5个的冷启动用户

        # TODO: 2.split the data into 70-10-20 based on the timestamp
        #根据时间戳将数据划分为70%-10%-20%

        #读取数据
        #df_train = pd.read_csv(self.path, sep=self.sep, names=self.header)
        '''
        if not num_users and not num_items:
            n_users = df_train.user.unique().shape[0]
            n_items = df_train.item.unique().shape[0]
        else:
            n_users = num_users #7390 #43117 #df_train.user.unique().shape[0]
            n_items = num_items #10159#26018 #df_train.item.unique().shape[0]

        print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
        '''
        #train_data = pd.DataFrame(df_train)

        self.user_ids = list()
        self.item_ids = list()

        train_row = []
        train_col = []
        train_rating = []

        #构建用户和物品的词典
        """
        for line in train_data.itertuples():
            self.user_ids.append(line[1])
            self.item_ids.append(line[2])
        """
        # 逐行读取数据，只读取clientIP和URL
        with open(self.path, 'r') as fin:
            for line in fin:
                #u, i, _ = line.strip().split()
                u, i, _ = line.strip().split()
                self.user_ids.append(u)
                self.item_ids.append(i)

        for u in self.user_ids:
            if u not in user_map:
                user_map[u] = self.num_user
                self.num_user += 1

        for i in self.item_ids:
            if i not in item_map:
                item_map[i] = self.num_item
                self.num_item += 1
        if num_users and num_items:
            self.num_user = num_users
            self.num_item = num_items
        print("....Load data finished. Number of users:", self.num_user, "Number of items:", self.num_item)
        self.user_map = user_map
        self.item_map = item_map

        #利用词典将用户和物品序列转化为ID
        self.user_ids = np.array([self.user_map[u] for u in self.user_ids])
        self.item_ids = np.array([self.item_map[i] for i in self.item_ids])
        print(len(self.item_ids))

        if isTrain:
            #加载数据序列
            self.load_data_seq()
            # else:
            #    self.num_item += 1

    def load_data_seq(self):
        # 为了对齐序列，而将URL ID从1开始
        for k, v in self.item_map.items():
            self.item_map[k] = v + 1
        self.item_ids = self.item_ids + 1
        self.num_item += 1

        #最长序列的长度
        max_seq_len = self.seq_len + self.target_len

        # 使用用户IP的ID 对原始样本进行排序
        #numpy.lexsort() 用于对多个序列进行排序。把它想象成对电子表格进行排序，每一列代表一个序列，排序时优先照顾靠后的列。
        sort_indices = np.lexsort((self.user_ids,))

        u_ids = self.user_ids[sort_indices]
        i_ids = self.item_ids[sort_indices]

        #unique（）保留数组中不同的值，返回两个参数（元素，元素出现的起始位置， 出现的次数）
        u_ids, indices, counts = np.unique(u_ids, return_index=True, return_counts=True)
        print ("counts:", counts)

        #计算每个用户交互物品序列的可分割样本数，并求和
        num_subsequences = sum([c - max_seq_len + 1 if c >= max_seq_len else 1 for c in counts])

        sequences = np.zeros((num_subsequences, self.seq_len))
        sequences_targets = np.zeros((num_subsequences, self.target_len))

        sequence_users = np.empty(num_subsequences)

        test_sequences = np.zeros((self.num_user, self.seq_len))
        test_users = np.empty(self.num_user)

        _uid = None
        # print(u_ids)
        # print(len(i_ids))

        #产生分割的序列
        for i, (uid, item_seq) in enumerate(self._generate_sequences(u_ids, i_ids, indices, max_seq_len)):
            if uid != _uid:
                print ("item_seq:", item_seq)
                test_sequences[uid][:] = item_seq[-self.seq_len:]
                test_users[uid] = uid
                _uid = uid
            #对于第i个样本，分别从物品序列中切分出标签和特征，标签在后半段，特征在前半段
            sequences_targets[i][:] = item_seq[-self.target_len:]
            sequences[i][:] = item_seq[:self.seq_len]
            sequence_users[i] = uid

        #对数据进行打包
        self.sequences = self.SeqData(sequence_users, sequences, sequences_targets)
        self.test_sequences = self.SeqData(test_users, test_sequences)

    # user_seq = []
    # for i in range(len(indices)):
    #     start_idx = indices[1]
    #
    #     if i >= len(indices) - 1:
    #         stop_idx = None
    #     else:
    #         stop_idx = indices[ i + 1]
    #
    #     # seq = []
    #     tensor = i_ids[start_idx:stop_idx]
    #     if len(tensor) - max_seq_len >= 0:
    #         for j in range(len(tensor), 0, -1):
    #             if j - max_seq_len >= 0:
    #                 user_seq.append((u_ids[i], tensor[j - max_seq_len:j]))
    #             else:
    #                 break
    #     else:
    #         user_seq.append((u_ids[i],tensor))
    #
    # _uid = None
    # for i, (uid, item_seq) in enumerate(user_seq):
    #     if uid != _uid:
    #         test_sequences[uid][:] = item_seq[-sequence_len:]
    #         test_users[uid] = uid
    #         _uid = uid
    #     sequence_targets[i][:] = item_seq[-target_len:]
    #     sequences[i][:] = item_seq[:sequence_len]
    #     sequence_users[i] = uid



    def _sliding_window(self, tensor, window_size, step_size=1):
        if len(tensor) - window_size >= 0:
            #在长度len的Tensor中，可分割样本数为len-step_size
            for i in range(len(tensor), 0, -step_size):
                if i - window_size >= 0:
                    yield tensor[i - window_size:i]
                else:
                    break
        else:
            yield tensor

    def _generate_sequences(self, user_ids, item_ids,
                            indices,
                            max_sequence_length):
        for i in range(len(indices)):
            #indices为某个use_id的起始位置
            start_idx = indices[i]

            if i >= len(indices) - 1:
                stop_idx = None
            else:
                #下一个use_id的起始位置可以作为上一个user_id的终止位置
                stop_idx = indices[i + 1]

            #对物品序列进行滑窗分割
            for seq in self._sliding_window(item_ids[start_idx:stop_idx],
                                            max_sequence_length):
                #返回用户ID和分割的序列
                yield (user_ids[i], seq)

    def tocsr(self):

        row = self.user_ids
        col = self.item_ids
        data = np.ones(len(row))

        return csr_matrix((data, (row, col)), shape=(self.num_user, self.num_item))





        #
        # for u in range(n_users):
        #     for i in range(n_items):
        #         train_row.append(u)
        #         train_col.append(i)
        #         if (u, i) in train_dict.keys():
        #             train_rating.append(1)
        #         else:
        #             train_rating.append(0)
        #
        # all_items = set(np.arange(n_items))
        #
        # neg_items = {}
        # train_interaction_matrix = []
        # for u in range(n_users):
        #     neg_items[u] = list(all_items - set(train_matrix.getrow(u).nonzero()[1]))
        #     train_interaction_matrix.append(list(train_matrix.getrow(u).toarray()[0]))
        #
        # test_row = []
        # test_col = []
        # test_rating = []
        # for line in test_data.itertuples():
        #     test_row.append(line[1] - 1)
        #     test_col.append(line[2] - 1)
        #     test_rating.append(1)
        # test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))
        #
        # test_dict = {}
        # for u in range(n_users):
        #     test_dict[u] = test_matrix.getrow(u).nonzero()[1]
        #
        # print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
        #
        #
        #
        # num_subsequences = sum([ c - max])
        #
        # return train_interaction_matrix, test_dict, n_users, n_items

# def remove_cold_start_user():
#     print()
#
#
# def train_valid_test_split(time_order=True):
#     # if time_order:
#
#     print()

# if __name__ == '__main__':
#

def data_preprocess(path, path_save, sep="\t", header = ['user_id', 'item_id', 'rating', 'timestampe']):

    #TODO: leave the recent one for test, seperately the data into two parts.
    df = pd.read_csv(path, sep=sep, names=header, engine='python')
    test_items = {}
    n_users = df.user_id.unique().shape[0]  # 943  # 6040 #.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]  # 1682 # 3952 ##df.item_id.unique().shape[0]
    print("Number of users: %d; Number of items: %d;" % (n_users, n_items))
    train_items = {}
    user_set = set()
    for line in df.itertuples():
        u = line[1]
        i = line[2]
        user_set.add(u)
        train_items.setdefault(u, []).append((u, i, line[3],line[4]))
        if u not in test_items:
            test_items[u] = (i, line[3], line[4])
        else:
            if test_items[u][2] < line[4]:
                test_items[u] = (i, line[3], line[4])


    test_data = [(key, value[0], value[1],  value[2]) for key, value in test_items.items()]
    test_data_map = {}
    for i in range(len(test_data)):
        test_data_map[test_data[i][0]] = test_data[i]

    test_file = open(path_save+"test.dat", 'a', encoding='utf-8')
    test_writer = csv.writer(test_file, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
    for i in test_data:
        #test_writer.writerow([i[0] - 1 , i[1]-1 ,   i[2]])
        test_writer.writerow([i[0]  , i[1] , i[2],  i[3]])

    train_file = open(path_save+"train.dat", 'a', encoding='utf-8')
    train_writer = csv.writer(train_file, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)

    for u in user_set:
        sorted_items = sorted(train_items[u ], key=lambda tup: tup[3], reverse=False)
        #print(sorted_items)

        for i in sorted_items:
            #print(test_data[u])
            #print(sorted_items[i])
            #print(u)
            if i != test_data_map[u]:
                train_writer.writerow([u , i[1], i[2], i[3]])
