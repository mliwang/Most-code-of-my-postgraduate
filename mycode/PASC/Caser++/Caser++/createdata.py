#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time:2019/10/10 1:26
#@Author:Lujianmin 

import pandas as pd

alldata = pd.read_csv("processed.csv", sep=',', header=0, usecols=[0,1,4])
alldata.columns = ['Url', 'Time', 'ClientIP']
b=alldata.sort_values(by="Time",ascending=True) #升序排列 时间靠前的在前面
splitTime=848284875121897.5
train=b.loc[b["Time"]<=splitTime]
test=b.loc[b["Time"]>splitTime]
# print(train)
# print(test)
train_user=set(train["ClientIP"])
test_user=set(test["ClientIP"])
concernUser=train_user&test_user
#只保留两者公共用户的记录
test=test.loc[test["ClientIP"].isin(concernUser)]
train=train.loc[train["ClientIP"].isin(concernUser)] #
# print(train)  19158
# print(test) 19483
#把只有一个序列的用户删除 先从训练集中找到只有一个序列的用户，然后记下用户id 再删除他在测试集中的记录即可
# testdict=dict(test.groupby(["clientIP"]).URL.unique().map(lambda x:list(x)))
traindict=dict(train.groupby(["ClientIP"]).Url.unique().map(lambda x:list(x)))
# print(traindict)  有一个所有url都相同的用户也删除掉了
clientsIP = set()
for k,v in traindict.items():
    if len(v)<=1:
        # print(k)
        clientsIP.add(k)
# print(clientsIP)
train=train.loc[~train["ClientIP"].isin(clientsIP)]
test=test.loc[~test["ClientIP"].isin(clientsIP)]
# print(test) 19271
# print(train) 19143
train['rating']='1'
test['rating']='1'
# # print(test)
train_dataset=list(zip(train['ClientIP'],train['Url'],train['rating']))
test_dataset=list(zip(test['ClientIP'],test['Url'],test['rating']))
d = pd.DataFrame(test_dataset)
d.to_csv('test.txt', sep='\t',index=False, header=None)
# print(test_dataset)
# print(train_dataset)
