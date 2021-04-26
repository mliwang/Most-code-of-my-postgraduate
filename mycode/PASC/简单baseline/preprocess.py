# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:00:16 2019

@author: Administrator
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from CCF1 import CF_svd,CF_knearest,CF_user,M_F
import Cachepolice as Ca
#trace=pd.read_csv("trace_detail_8482",header =None,sep=" ")
#trace.columns=["req_time","firstByteTime","LastByteTime","clientIP","ServerIP","clientHeader",
#"ServerHeader","IfModefinedSinceClientHeader","ExpiresServerHeader", "LastModefiedServerHeader","responseHeaderLen",
#"responseDataLan","URLLen", "GET", "URLValue", "HTTP"]

def readdata(path):
    size_sum = 0
    max_size = 0
    supported = 0
    timestamps = []
    urls = set()
    clientsIP = set()
    serversIP = set()
    url_size= defaultdict()
    eventList = []
    f = open(path)
    print ('reading trace file ' + path)
    line = f.readline()
    while line:
        
        
            #fields = [req_time,firstByteTime,LastByteTime,clientIP,ServerIP,clientHeader,
            #ServerHeader,IfModefinedSinceClientHeader,ExpiresServerHeader, LastModefiedServerHeader,responseHeaderLen,
            #responseDataLan,URLLen, GET, URLValue, HTTP/1.0]
        fields = line.rstrip().split(" ")
            
            #if the request is not GET request deny it
        if len(fields)>=15 and fields[13]=='GET':
                #compute response time in microseconds
                #original_responseLen = (int(fields[11]) + int(fields[10]))*8
                #responseLen = self.generate_size(int(fields[10]),int(fields[11]),self.distribution)
            responseLen = (int(fields[10])+int(fields[11]))#数据包真正大小
            if responseLen>0:
                t = fields[0].split(":")
                start_dt = int(t[0])*1000000+int(t[1])
                t = fields[2].split(":")
                end_dt = int(t[0])*1000000+int(t[1])
    #                    t = fields[1].split(":")
    #                    first_byte = int(t[0])*1000000+int(t[1])
                responseTime = (end_dt - start_dt)/1000
                if responseTime==0:
                    responseTime = 1
                cip = fields[3].split(":")[0]
                sip = fields[4].split(":")[0]
                timestamps.append(start_dt/1000)
                clientsIP.add(cip)
                serversIP.add(sip)
                url = fields[14]
                urls.add(url)
                headLen = int(fields[10])
                url_size.setdefault(url,responseLen)
                if responseLen > url_size[url]:
                    url_size[url] = responseLen
                    
                if url.count("gif")>0 or url.count("jpg")>0 or url.count("jpeg")>0 or url.count("mp4")>0 or url.count("mov")>0 or url.count("mp3")>0 or url.count("swf")>0 \
                    or url.count("GIF")>0 or url.count("JPG")>0 or url.count("JPEG")>0 or url.count("MP4")>0 or url.count("MOV")>0 or url.count("MP3")>0 or url.count("SWF")>0 \
                    or url.count("exe")>0 or url.count("PNG")>0 or url.count("zip")>0 or url.count("ZIP")>0 or url.count("tar")>0 or url.count("rar")>0 or url.count("TAR")>0 \
                    or url.count("RAR")>0 or url.count("tar.gz")>0:
                    
                    isSupported = True
                    if max_size<responseLen:
                        max_size = responseLen
                    size_sum = size_sum + responseLen
                    supported = supported + 1
                else:
                    isSupported = False
                eventList.append(pd.Series({'URL':url,'timestamp':start_dt, 'latency':responseTime,\
                        'speed':responseLen/float(responseTime),'clientIP':cip,'serverIP':sip,\
                        'len':responseLen,'headLen':headLen,'isSupported':isSupported}))
           
        
        line = f.readline()
            #print (line )
        #sort eventList on request timestamp
    f.close()
    d=pd.DataFrame(eventList)
    d.to_csv("processed.csv",index=False)
    return eventList
#eventList=readdata("trace_detail_8482")
def findbetterSplit(b):
    mint,mean,threeQu,maxt=list(b.describe().loc[["min","mean","75%","max"],["timestamp"]].values.flatten())
    maxUser=0
    bettersplit=mean
    for i in list(np.arange(mint,maxt,(maxt-mint)/1000)):
        splitTime=i#划分点
        train=b.loc[b["timestamp"]<=splitTime]#从时间的中位数划分
        test=b.loc[b["timestamp"]>splitTime]
        train_user=set(train["clientIP"])
        test_user=set(test["clientIP"])
        concernUser=train_user & test_user
        if len(concernUser)>maxUser:
            maxUser=len(concernUser)
            bettersplit=splitTime
            
    return bettersplit,maxUser #在这两个时间段都有请求的用户数为201

def getMatrixP(t):#获取用户和内容的流行度矩阵
    user=list(set(t["clientIP"]))
    conttent=list(set(t["URL"]))
    P=np.zeros((len(user),len(conttent)))
    for i,r in t.iterrows():
        uid=user.index(r["clientIP"])
        urlc=conttent.index(r["URL"])
        P[uid,urlc]+=1
    P=pd.DataFrame(P,index=user,columns=conttent)
    return P    



def evaluate(predict,reldict,N):
    """评估函数
    predict  模型推荐结果，dict类型，key为 "clientIP"，value为"URL"list
    relldict  各个用户真实点击情况，dict类型，key为 "clientIP"，value为"URL"list
    N为给每个用户推荐的内容数目
    return NetworkLoadRate,CacheRaplaceRate
    NetworkLoadRate float类型， sum（len(x in Ri and not in Pi)）/sum(len(Ri))
    CacheRaplaceRate float类型， Avg((N-count(x in Ri and in Pi))/N)
    """
    n1=0.0
    n2=0.0
    C=[]
    
    for key, value in reldict.items():
        n2=n2+len(value)
        extrat_ask=len(set(value)-set(predict[key]))
        n1=n1+extrat_ask
#        print(len(set(value).intersection(set(predict[key]))))
        C.append((N-len(set(value).intersection(set(predict[key]))))/N)
    return n1/n2,sum(C)/len(C)

def getCFresult(K,strategy="user_CF"):
    alldata=pd.read_csv("processed.csv")
    b=alldata.sort_values(by="timestamp" , ascending=True) 
#    splitTime,maxUser=findbetterSplit(b)#划分点 848284875121897.5
#    print(splitTime)
    splitTime=848284875121897.5  #两个时间段内都有行为的用户个数为212
    train=b.loc[b["timestamp"]<=splitTime]#从时间的中位数划分
    test=b.loc[b["timestamp"]>splitTime]
    train_user=set(train["clientIP"])
    test_user=set(test["clientIP"])
    concernUser=list(train_user & test_user)
    #找到只含关心的用户的真实请求情况
    test=test.loc[test["clientIP"].isin(concernUser)]
    testdict=dict(test.groupby(["clientIP"]).URL.unique().map(lambda x:list(x)))
    
    trainP=getMatrixP(train)#拿到训练集的流行度矩阵，index为用户，columns为url
#    print(trainP)
    if strategy=="svd":
        cf = CF_svd(k=K, r=3)
        rec=cf.fit(trainP,concernUser)
#        train_dataFloat=trainP.values/255.0
        
        
    elif strategy=="user_CF":
        cf = CF_user(k=K)
        rec=cf.fit(trainP,concernUser)
    elif strategy=="MF":
        cf = M_F(trainP,K,concernUser)
        rec=cf.matrix_fac(0.0001,0.0002)     
        
    elif strategy=="item_CF":
        cf = CF_knearest(k=K)
        rec=cf.fit(trainP.T.iloc[:15000].T,concernUser)
    elif strategy=="LFU":
        rec=Ca.LFU(trainP,K,concernUser)
    elif strategy=="LRU":
        rec=Ca.LRU(train,K,concernUser)
    elif strategy=="FIFO":
        rec=Ca.FIFO(train,K,concernUser)   
    
    
    NetworkLoadRate,CacheRaplaceRate=evaluate(rec,testdict,K)    
#    print(NetworkLoadRate,CacheRaplaceRate)
    return rec,testdict,NetworkLoadRate,CacheRaplaceRate


fresultLarge = open('ResultWithMF.txt','w')
for k in range(100,1100,100):
    t,rel,NetworkLoadRate,CacheRaplaceRate=getCFresult(k,"MF")
    print("MF"," ",k," ",NetworkLoadRate," ",CacheRaplaceRate)
    print("MF"," ",k," ",NetworkLoadRate," ",CacheRaplaceRate,"\n",file=fresultLarge)
fresultLarge.close()

#import pickle
#def main():
#    polices=["LFU","LRU","FIFO","svd","user_CF","item_CF"]
##    polices=["item_CF"]
#    K=range(100,1100,100)
##    small_K=[1,50,100,150,200,250,300]
#    fresultLarge = open('wellResult.txt','w')  
#    print("The following lines represent Strategy CacheSize  NetworkLoadRate  CacheRaplaceRate\n",file=fresultLarge)
##    fresultsmall = open('item_CF_SmallScale.txt','w')  
##    print("The following lines represent Strategy CacheSize  NetworkLoadRate  CacheRaplaceRate",file=fresultsmall)
#    for p in polices:
#        for k in K:
#            t,rel,NetworkLoadRate,CacheRaplaceRate=getCFresult(k,p) 
#            print(p," ",k," ",NetworkLoadRate," ",CacheRaplaceRate)
#            print(p," ",k," ",NetworkLoadRate," ",CacheRaplaceRate,"\n",file=fresultLarge)
#            pickle.dump(t, open(str(p)+'_'+str(k)+'_.pickle', 'wb'))
##        for k in small_K:
##            t,rel,NetworkLoadRate,CacheRaplaceRate=getCFresult(k,p) 
##            print(p," ",k," ",NetworkLoadRate," ",CacheRaplaceRate)
##            print(p," ",k," ",NetworkLoadRate," ",CacheRaplaceRate,file=fresultsmall)
##            pickle.dump(t, open(p+'_'+str(k)+'_.pickle', 'wb'))
#        print("\n",file=fresultLarge)
#    fresultLarge.close()
##    fresultsmall.close()
#    return
#main()
#print("done！")
#    