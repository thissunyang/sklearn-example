import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt

mac2id = dict()
onlinetimes = []

f=open('TestData.txt', encoding='utf-8')
for line in f:    #print(line)  line 是每一行的数据
    items=line.strip().split(',')     # 把每一行返回成一个列表
    # print(items)
    mac=items[2]   #储存MAC地址
    onlinetime = int(items[6])  # 储存上网时长    1558
    starttime = int(items[4].split(' ')[1].split(':')[0])
    # print(items[4].split(' ')[1])   #输出以空格分隔的  第一项即：22:44:18.540000000
    # print(items[4].split(' ')[1].split(':')[0])#继续输出第一项之后  里面以：分隔的 第0项 即：22


    #保证onlinetime中对应一个mac地址有一个唯一的记录
    if mac not in mac2id:                      #如果是新的mac地址，那么记录下mac地址，开始上网时间和上网时长
        mac2id[mac] = len(onlinetimes)
    # print(mac2id)
        onlinetimes.append((starttime,onlinetime))
    # print(onlinetimes)
    else:                                      #如果mac重复，那么更新mac地址，开始上网时间和上网时长
        mac2id[mac] = len(onlinetimes)
        onlinetimes[mac2id[mac]]=(starttime,onlinetime)
# print(len(onlinetimes))
real_X=np.array(onlinetimes).reshape(-1,2)
# print(real_X)
X=real_X[:, 0:1] # X=real_X[:,1:2]  取出上网时长
print(X)
db=DBSCAN(eps=0.01, min_samples=20).fit(X)   #调用DBCAN进行训练
labels = db.labels_
print(labels)                                #输出，每个数据的标签

# raito=len(labels[labels[:] == -1])/len(labels)   #计算标签为-1的数据（即噪声数据)的比例
# # print('Noise raito:',format(raito, '.2%'))   #Noise raito: 22.15%
n_clusters_=len(set(labels))-(1 if -1 in labels else 0)  # 如果里面含有噪声(-1)，则把噪声删除
print('Estimated number of clusters:%d'%n_clusters_)
'''
评价聚类效果:轮廓系数s(i)
    s(i) 越接近于1  说明样本聚类合理
    s(i) 越接近于0  说明样本i在两个簇的边界上。
'''
# print("Silhouette Coefficient:%0.3f" % metrics.silhouette_score(X,labels))
for i in range(n_clusters_):
    print('number of data in Cluster %s is:%s'%(i,len(X[labels==i])))

plt.hist(X, 24)
plt.show()