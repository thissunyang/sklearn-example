import numpy as np
from sklearn.cluster import KMeans

def loadData(filePath):
    fr = open(filePath, 'r+', encoding = 'utf-8') #打开文件并增加读写
    lines = fr.readlines()   #每次读取整个文件保存在一个list中，list中的每个元素为文件的每一行数据（字符串类型）
    retCityName=[]           #定义列表储存  城市的名字         赋给cityName
    retData=[]               #定义列表储存  城市的各项消费信息  赋给data
    for line in lines:
        items = line.strip().split(",")   # 把每一行返回成一个列表
        retCityName.append(items[0])      # 把每一行的第一个元素(城市的名字)填充到retCityName列表中
        retData.append([float(items[i]) for i in range(1, len(items))])
    return retData,retCityName

if __name__=='__main__':
    data, cityName = loadData('city.txt')
    print(cityName)
    km=KMeans(n_clusters=4)#聚类中心为4；可修改
    label=km.fit_predict(data)#label对应每行数据对应分配到的序列,相同的序号归为一类
    print(label)
    print('km.cluster_centers_\n',km.cluster_centers_)
    # 打印 归为同一个簇的城市  每一项花费的平均值，共有4*8列，比如3242.22333333=（2959.19+3712.31+3712.31）/3，其他的类似

    expenses=np.sum(km.cluster_centers_,axis=1)         # 求和 对每一行求和  1*4
    # print('expenses\n',expenses,'\n\n')
    CityCluster=[[],[],[],[]]       # 定义四个簇的   空列表
    for i in range(len(cityName)):  # 31个城市
       CityCluster[label[i]].append(cityName[i])#把每个 城市根据  标签  写进  对应的簇里面

    for i in range(len(CityCluster)):            #打印输出每一个簇
       print("expenses:%.2f" % expenses[i])     #平均花费格式化输出
       print(CityCluster[i])                    #每个簇的城市名称