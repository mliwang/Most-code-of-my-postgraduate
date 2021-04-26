#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time:2019/10/12 16:44
#@Author:Lujianmin
import numpy as np
from gensim.models import word2vec
from gensim.models import FastText

# test_sequences = np.zeros((5,5),dtype=np.int64)
# test_sequences[0][:]=[1,2,3,4,5]
# print(test_sequences)

# 生成word2vec词向量 128维
import gensim
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
import multiprocessing
from random import random

path = get_tmpfile("/home/kesci/word2vec.model")  # 创建临时文件

sentences = []  # 句子的list
f = open("/home/kesci/input/bytedance/train_final.csv")
for line in f:
    line = line.strip().split(",")
    if random() < 0.1:  # 以0.1的概率随机抽取句子title和query
        sentences.append(line[1])
        sentences.append(line[3])
f.close()

f = open("/home/kesci/input/bytedance/test_final_part1.csv")
for line in f:
    line = line.strip().split(",")
    sentences.append(line[1])
    sentences.append(line[3])
f.close()

sentences = []
f = open("/home/kesci/test2.csv")
for line in f:
    line = line.strip().split(",")
    sentences.append(line[1])
    sentences.append(line[3])
f.close()
print(len(sentences))


class MySentences(object):
    def __init__(self, sentences):
        self.sentences = sentences

    def __iter__(self):
        print("djkbakjs")
        for line in self.sentences:
            yield line.split()


all_sentences = MySentences(sentences)
model = Word2Vec(all_sentences, size=128, window=5, min_count=5, workers=4)
model.save("/home/kesci/word2vec.model")

# 生成FastText词向量 128维
from gensim.models import FastText

path = get_tmpfile("./data/fast_w2v.model")  # 创建临时文件


class MySentences(object):
    def __init__(self, sentences):
        self.sentences = sentences

    def __iter__(self):
        print("djkbakjs")
        for line in self.sentences:
            yield line.split()


all_sentences = MySentences(sentences)
model = FastText(all_sentences, size=128, window=5, min_count=5, workers=4)
model.save("./data/fast_w2v.model")

# 生成word2id文件，统计每个词的词频，从大到小排序之后，并且去除词频小于5的词
my_dict = {}
for ju in sentences:
    for w in ju.split():
        my_dict.setdefault(w, 0)
        my_dict[w] += 1
p_d = sorted(my_dict.items(), key=lambda item: item[1], reverse=True)
f_w = open("./data/word2id.txt", "w")
idx = 1
sum_word = 0
for i in p_d:
    if i[1] >= 5:
        sum_word += 1
        f_w.write(i[0] + " " + str(idx) + "\n")
        idx += 1
print(sum_word)
f_w.close()

word2id_file="./data/word2id.txt"
read_file = open(word2id_file, "r")
word2id={}
for i in read_file:
    i=i.strip().split()
    word2id[i[0]]=int(i[1])
read_file.close()
# /////////////////////
#参数配置
epcho=1
batch_size=256
num_to_ev=400 # 训练多少批，在本地评测一次
vocab_size=len(word2id) # 词典大小
embedding_dim=256 # 词向量维度
t_max_len=22 #title的最大长度
q_max_len=11 #query的最大长度
lr=0.0001 #学习率

ce = np.random.uniform(-1, 1, [vocab_size + 1,embedding_dim])
word2vec_model = Word2Vec.load("/home/kesci/word2vec.model")
fast_model = FastText.load("./data/fast_w2v.model")
ce[0] = np.zeros(embedding_dim)
for i in word2id:
    try:
        ce[word2id[i]] = np.concatenate((word2vec_model[i],fast_model[i]))
    except:
        print(i)

