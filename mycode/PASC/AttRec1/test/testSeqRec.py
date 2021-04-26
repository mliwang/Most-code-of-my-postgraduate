import argparse
import tensorflow as tf
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#from models.seq_rec.Caser import Caser
from models.seq_rec.AttRec import AttRec
#from models.seq_rec.PRME import PRME
from utils.load_data.load_data_seq import DataSet
#from utils.load_data.load_data_ranking import *


def parse_args():
    parser = argparse.ArgumentParser(description='DeepRec')
    parser.add_argument('--model', choices=['Caser','PRME', 'AttRec'], default = 'AttRec')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_factors', type=int, default=100)
    parser.add_argument('--display_step', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128 ) #128 for unlimpair1024
    parser.add_argument('--learning_rate', type=float, default=1e-3) #1e-4 for unlimpair
    parser.add_argument('--reg_rate', type=float, default=0.1) #0.01 for unlimpair
    return parser.parse_args()


if __name__ == '__main__':
    #参数配置
    args = parse_args()
    epochs = args.epochs
    learning_rate = args.learning_rate
    reg_rate = args.reg_rate
    num_factors = args.num_factors
    display_step = args.display_step
    batch_size = args.batch_size


    #GPU训练配置
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True



    with tf.Session(config=config) as sess:
        model = None

        if args.model == "AttRec":
            #读入训练和测试数据
            train_data = DataSet(path="../train.txt", sep="\t",header=['user', 'item', 'rating', 'time'],isTrain=True, seq_len=3, target_len=1, num_users=0, num_items=0)
            test_data = DataSet(path="../test.txt", sep="\t", header=['user', 'item', 'rating', 'time'], user_map=train_data.user_map, item_map=train_data.item_map)
            
            #初始化模型
            model = AttRec(sess, train_data.num_user,  train_data.num_item)
            # print(train_data.user_map)
            # print(train_data.item_map)

            #构建模型，输入马尔科夫链的阶数，和目标的长度
            model.build_network(L = train_data.sequences.L, num_T=train_data.sequences.T)

            #执行模型
            model.execute(train_data, test_data)
