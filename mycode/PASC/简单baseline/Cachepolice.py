# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 16:20:16 2019

@author: Administrator
"""
import pandas as pd
import numpy as np
import random
def getPoplarity(P):
    sum_popularity= P.apply(lambda x: x.sum())
    sum_popularity=list(sum_popularity.sort_values(ascending=False).index)#拿到内容的候选列表
    return sum_popularity
    
def LFU(P,N,predict):
    """
    P为用户点击频率矩阵
    N为最终每个用户需要留存的内容数目
    predict为需要关注的用户
    """
    rec={}
#    sum_popularity= getPoplarity(P)
    
    for u in predict:
        u_c=P.loc[u]#该用户当前缓存情况
        u_c=u_c[u_c!=0]
        u_c=u_c[u_c!=1]
        u_c=u_c.sort_values(ascending=False)#按过去点击频率获取将来可能点击的列表   
        if(len(u_c)>=N):
            rec[u]=list(u_c.index)[:N]
        else:
#            reclist=list(u_c.index)
#            recom=list(P.columns)
#            random.shuffle(recom)
#            for pc in recom:
#                if(len(reclist)>=N):
#                    rec[u]=reclist[:N]
#                    break
#                if pc not in reclist:
#                    reclist.append(pc)
            #不能随机给，随机给会使效果提升
            reclist=list(u_c.index)
            reclist.extend(["##"]*(N-len(reclist)))
            rec[u]=reclist[:N]
#    
    return rec
def LRU(train,N,predict):
    """
    train为用户点击序列
    N为最终每个用户需要留存的内容数目
    predict为需要关注的用户
    """
    dd=dict(train.groupby(["clientIP"]).apply(lambda x:list(x.sort_values(by="timestamp" , ascending=True)["URL"])))#每个用户最近点击内容的序列
    rec={}
#    sum_popularity= getPoplarity(P)
    
    for u in predict:
        u_c=dd[u]#该用户当前缓存情况
        cache=[]
        #开始淘汰
        for c in u_c:
            if len(cache)<N:
                if c not in cache:
                    cache.append(c) 
                else:
                    cache.append(c)
                    cache=cache[1:]
            else:
                if c not in cache:
                    cache.append(c)
                    cache=cache[1:]
                else:
                    cache.remove(c)
                    cache.append(c)
        
        if(len(u_c)>=N):
            rec[u]=cache
            
        else:
#            recom=list(train["URL"].values)
#            random.shuffle(recom)
#            for pc in recom:
#                if(len(cache)>=N):
#                    rec[u]=cache[:N]
#                    break
#                if pc not in cache:
#                    cache.append(pc)
            cache.extend(["##"]*(N-len(cache)))
            rec[u]=cache[:N]
    
    return rec
def FIFO(train,N,predict):
    """
    train为用户点击序列
    N为最终每个用户需要留存的内容数目
    predict为需要关注的用户
    """
    dd=dict(train.groupby(["clientIP"]).apply(lambda x:list(x.sort_values(by="timestamp" , ascending=True)["URL"])))#每个用户最近点击内容的序列
    rec={}
#    sum_popularity= getPoplarity(P)
    for u in predict:
        u_c=dd[u]#该用户点击历史情况
        cache=[]
        for c in u_c[::-1]:
            if c not in cache:
                cache.append(c)
        if(len(cache)>=N):
            rec[u]=cache[:N]
        else:
#            recom=list(train["URL"].values)
#            random.shuffle(recom)
#            for pc in recom:
#                if(len(cache)>=N):
#                    rec[u]=cache[:N]
#                    break
#                if pc not in cache:
#                    cache.append(pc)
            cache.extend(["##"]*(N-len(cache)))
            rec[u]=cache[:N]
    
    return rec