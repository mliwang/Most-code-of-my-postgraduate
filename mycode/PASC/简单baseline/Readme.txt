1.preprocess.py 入口，功能：trace data的预处理、测试集训练集最好划分点（测试集和训练集公共user尽量多）、获取用户和内容的流行度矩阵、评估函数、拿到各种推荐方式的推荐结果并把结果记录入txt.
2.Cachepolice.py传统缓存策略实现
3.CCF1.py 各种推荐策略实现
4.processed.csv  trace data预处理后的数据每行对应一条跟踪数据
5.trace_detail_8482 原始跟踪数据，数据来源与说明参考PICN论文 与https://github.com/zzali/PICN-simulation
6.wellResult.txt 为实验结果，matlib中为绘图最终实验结果